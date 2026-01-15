"""
Adaptive Risk Controller

Automatically adjusts risk settings (shorting, leverage, aggressive mode)
based on market conditions to maximize gains and protect against losses.

The system:
1. Monitors market regime, volatility, and trend
2. Decides when to enable/disable risky features
3. Logs all decisions with reasoning
4. Shows current strategy on dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Risk profile levels."""
    CONSERVATIVE = "conservative"  # No shorting, no leverage, safe strategies
    MODERATE = "moderate"          # Some shorting allowed, no leverage
    AGGRESSIVE = "aggressive"      # Shorting + leverage allowed
    MAXIMUM = "maximum"            # All features enabled for max opportunity


@dataclass
class RiskDecision:
    """A risk adjustment decision."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # What changed
    setting: str = ""  # "shorting", "leverage", "aggressive"
    old_value: bool = False
    new_value: bool = False

    # Why
    reason: str = ""
    market_regime: str = ""
    trigger_condition: str = ""

    # Context
    rsi: float = 50.0
    volatility: str = "normal"
    trend: str = "neutral"
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "setting": self.setting,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "market_regime": self.market_regime,
            "trigger_condition": self.trigger_condition,
            "context": {
                "rsi": self.rsi,
                "volatility": self.volatility,
                "trend": self.trend,
                "confidence": self.confidence,
            }
        }


@dataclass
class CurrentStrategy:
    """Current active strategy for display."""
    name: str = "Conservative Hold"
    description: str = "Waiting for clear opportunity"

    # Risk settings
    shorting_enabled: bool = False
    leverage_enabled: bool = False
    aggressive_enabled: bool = False

    # Market view
    market_regime: str = "unknown"
    expected_direction: str = "neutral"  # bullish, bearish, neutral
    confidence: float = 0.0

    # Position guidance
    suggested_action: str = "hold"
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0

    # Reasoning
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class AdaptiveRiskController:
    """
    Controls risk settings adaptively based on market conditions.

    Rules:
    - Enable SHORTING when: Bear regime + high confidence + RSI > 60
    - Enable LEVERAGE when: Strong trend + low volatility + high win rate
    - Enable AGGRESSIVE when: Clear regime + high momentum + system performing well
    - Disable all when: Crash/unknown regime or high drawdown
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        api_base_url: str = "http://localhost:8000",
    ):
        self.data_dir = data_dir
        self.api_base_url = api_base_url

        # Current state
        self.current_settings = {
            "shorting": False,
            "leverage": False,
            "aggressive": False,
        }

        self.current_strategy = CurrentStrategy()
        self.current_profile = RiskProfile.CONSERVATIVE

        # Decision history
        self.decision_history: List[RiskDecision] = []
        self.max_history = 100

        # Performance tracking
        self.recent_win_rate: float = 0.5
        self.recent_pnl: float = 0.0
        self.current_drawdown: float = 0.0

        # Load decision log
        self.decision_log_path = data_dir / "risk_decisions.json"
        self._load_history()

        logger.info("Adaptive Risk Controller initialized")

    def _load_history(self):
        """Load decision history from file."""
        try:
            if self.decision_log_path.exists():
                with open(self.decision_log_path) as f:
                    data = json.load(f)
                    # Just load recent history summary
                    self.decision_history = []
        except Exception as e:
            logger.warning(f"Could not load risk decision history: {e}")

    def _save_decision(self, decision: RiskDecision):
        """Save decision to history."""
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]

        # Save to file
        try:
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            history_dicts = [d.to_dict() for d in self.decision_history[-20:]]
            with open(self.decision_log_path, "w") as f:
                json.dump({"recent_decisions": history_dicts}, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save risk decision: {e}")

    async def update_risk_settings(self, settings: Dict[str, bool]) -> bool:
        """Update risk settings via API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/api/trading/risk-settings",
                    json=settings,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        self.current_settings.update(settings)
                        return True
                    else:
                        logger.error(f"Failed to update risk settings: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error updating risk settings: {e}")
            return False

    async def evaluate_and_adjust(
        self,
        market_regime: str,
        regime_confidence: float,
        rsi: float,
        volatility: str,  # "low", "normal", "high", "extreme"
        trend: str,  # "up", "down", "neutral"
        recent_performance: Optional[Dict] = None,
    ) -> CurrentStrategy:
        """
        Evaluate market conditions and adjust risk settings.

        Returns the current strategy for display.
        """
        changes_made = []
        reasoning = []

        # Update performance metrics if provided
        if recent_performance:
            self.recent_win_rate = recent_performance.get("win_rate", 0.5)
            self.recent_pnl = recent_performance.get("total_pnl", 0.0)
            self.current_drawdown = recent_performance.get("drawdown", 0.0)

        # =====================================================================
        # SHORTING DECISIONS
        # =====================================================================
        should_short = False
        short_reason = ""

        if market_regime in ("strong_bear", "bear", "crash"):
            if regime_confidence > 0.6 and rsi > 55:
                should_short = True
                short_reason = f"Bear market ({market_regime}) with RSI={rsi:.0f} - shorting opportunities"
                reasoning.append(f"SHORTING ENABLED: {short_reason}")

        elif market_regime == "sideways" and rsi > 70:
            should_short = True
            short_reason = "Overbought in sideways market - mean reversion short"
            reasoning.append(f"SHORTING ENABLED: {short_reason}")

        # Disable shorting in bull markets or high volatility
        if market_regime in ("strong_bull", "bull"):
            should_short = False
            if self.current_settings["shorting"]:
                reasoning.append("SHORTING DISABLED: Bull market - avoid shorting against trend")

        if volatility == "extreme" and should_short:
            should_short = False
            reasoning.append("SHORTING DISABLED: Extreme volatility - too risky")

        # Apply shorting change
        if should_short != self.current_settings["shorting"]:
            decision = RiskDecision(
                setting="shorting",
                old_value=self.current_settings["shorting"],
                new_value=should_short,
                reason=short_reason or "Market conditions changed",
                market_regime=market_regime,
                trigger_condition=f"RSI={rsi:.0f}, Vol={volatility}",
                rsi=rsi,
                volatility=volatility,
                trend=trend,
                confidence=regime_confidence,
            )
            changes_made.append(("shorting", should_short))
            self._save_decision(decision)

        # =====================================================================
        # LEVERAGE DECISIONS
        # =====================================================================
        should_leverage = False
        leverage_reason = ""

        # Only enable leverage in strong trends with good performance
        if market_regime in ("strong_bull", "strong_bear"):
            if volatility in ("low", "normal") and self.recent_win_rate > 0.55:
                if self.current_drawdown < 10:  # Not in drawdown
                    should_leverage = True
                    leverage_reason = f"Strong trend with {self.recent_win_rate:.0%} win rate - leverage for gains"
                    reasoning.append(f"LEVERAGE ENABLED: {leverage_reason}")

        # Disable leverage in risky conditions
        if volatility in ("high", "extreme"):
            should_leverage = False
            if self.current_settings["leverage"]:
                reasoning.append("LEVERAGE DISABLED: High volatility - protecting capital")

        if self.current_drawdown > 15:
            should_leverage = False
            if self.current_settings["leverage"]:
                reasoning.append(f"LEVERAGE DISABLED: In {self.current_drawdown:.1f}% drawdown - reducing risk")

        if market_regime in ("crash", "unknown"):
            should_leverage = False
            if self.current_settings["leverage"]:
                reasoning.append("LEVERAGE DISABLED: Uncertain/crash regime")

        # Apply leverage change
        if should_leverage != self.current_settings["leverage"]:
            decision = RiskDecision(
                setting="leverage",
                old_value=self.current_settings["leverage"],
                new_value=should_leverage,
                reason=leverage_reason or "Risk conditions changed",
                market_regime=market_regime,
                trigger_condition=f"WinRate={self.recent_win_rate:.0%}, DD={self.current_drawdown:.1f}%",
                rsi=rsi,
                volatility=volatility,
                trend=trend,
                confidence=regime_confidence,
            )
            changes_made.append(("leverage", should_leverage))
            self._save_decision(decision)

        # =====================================================================
        # AGGRESSIVE MODE DECISIONS
        # =====================================================================
        should_aggressive = False
        aggressive_reason = ""

        # Enable aggressive in strong trending markets with proven performance
        if market_regime in ("strong_bull", "strong_bear"):
            if regime_confidence > 0.7 and self.recent_win_rate > 0.6:
                if self.recent_pnl > 0 and self.current_drawdown < 5:
                    should_aggressive = True
                    aggressive_reason = f"High confidence {market_regime} + {self.recent_win_rate:.0%} win rate"
                    reasoning.append(f"AGGRESSIVE ENABLED: {aggressive_reason}")

        # Aggressive for momentum plays
        if abs(rsi - 50) > 30 and volatility == "low" and trend != "neutral":
            if self.recent_win_rate > 0.55:
                should_aggressive = True
                aggressive_reason = f"Strong momentum (RSI={rsi:.0f}) in low volatility"
                reasoning.append(f"AGGRESSIVE ENABLED: {aggressive_reason}")

        # Disable aggressive in uncertain conditions
        if market_regime in ("sideways", "unknown", "crash"):
            should_aggressive = False
            if self.current_settings["aggressive"]:
                reasoning.append(f"AGGRESSIVE DISABLED: {market_regime} regime - playing safe")

        if self.current_drawdown > 10:
            should_aggressive = False
            if self.current_settings["aggressive"]:
                reasoning.append("AGGRESSIVE DISABLED: In drawdown - capital protection mode")

        # Apply aggressive change
        if should_aggressive != self.current_settings["aggressive"]:
            decision = RiskDecision(
                setting="aggressive",
                old_value=self.current_settings["aggressive"],
                new_value=should_aggressive,
                reason=aggressive_reason or "Market conditions changed",
                market_regime=market_regime,
                trigger_condition=f"Confidence={regime_confidence:.0%}, PnL={self.recent_pnl:+.1f}%",
                rsi=rsi,
                volatility=volatility,
                trend=trend,
                confidence=regime_confidence,
            )
            changes_made.append(("aggressive", should_aggressive))
            self._save_decision(decision)

        # =====================================================================
        # APPLY CHANGES
        # =====================================================================
        if changes_made:
            new_settings = {k: v for k, v in changes_made}
            success = await self.update_risk_settings(new_settings)
            if success:
                logger.info(f"Risk settings updated: {new_settings}")
            else:
                logger.error("Failed to apply risk setting changes")

        # =====================================================================
        # BUILD CURRENT STRATEGY SUMMARY
        # =====================================================================
        self.current_settings["shorting"] = should_short
        self.current_settings["leverage"] = should_leverage
        self.current_settings["aggressive"] = should_aggressive

        # Determine strategy name
        strategy_name = self._determine_strategy_name(
            market_regime, should_short, should_leverage, should_aggressive
        )

        # Determine expected direction
        if market_regime in ("strong_bull", "bull"):
            expected_direction = "bullish"
            suggested_action = "buy" if rsi < 70 else "hold"
        elif market_regime in ("strong_bear", "bear", "crash"):
            expected_direction = "bearish"
            suggested_action = "sell" if should_short else "hold"
        else:
            expected_direction = "neutral"
            suggested_action = "hold"

        # Position sizing based on conditions
        if volatility == "extreme":
            position_multiplier = 0.3
        elif volatility == "high":
            position_multiplier = 0.5
        elif should_aggressive:
            position_multiplier = 1.5
        elif should_leverage:
            position_multiplier = 1.3
        else:
            position_multiplier = 1.0

        # Build strategy object
        self.current_strategy = CurrentStrategy(
            name=strategy_name,
            description=self._get_strategy_description(
                market_regime, expected_direction, suggested_action
            ),
            shorting_enabled=should_short,
            leverage_enabled=should_leverage,
            aggressive_enabled=should_aggressive,
            market_regime=market_regime,
            expected_direction=expected_direction,
            confidence=regime_confidence,
            suggested_action=suggested_action,
            position_size_multiplier=position_multiplier,
            stop_loss_multiplier=1.5 if volatility in ("high", "extreme") else 1.0,
            reasoning=reasoning if reasoning else ["Monitoring market conditions..."],
        )

        # Determine profile
        if should_aggressive and should_leverage and should_short:
            self.current_profile = RiskProfile.MAXIMUM
        elif should_aggressive or should_leverage:
            self.current_profile = RiskProfile.AGGRESSIVE
        elif should_short:
            self.current_profile = RiskProfile.MODERATE
        else:
            self.current_profile = RiskProfile.CONSERVATIVE

        return self.current_strategy

    def _determine_strategy_name(
        self,
        regime: str,
        shorting: bool,
        leverage: bool,
        aggressive: bool
    ) -> str:
        """Determine human-readable strategy name."""
        if regime in ("strong_bull", "bull"):
            if aggressive and leverage:
                return "Aggressive Bull Ride"
            elif aggressive:
                return "Momentum Long"
            else:
                return "Trend Following Long"

        elif regime in ("strong_bear", "bear"):
            if shorting and aggressive:
                return "Aggressive Bear Attack"
            elif shorting:
                return "Trend Following Short"
            else:
                return "Defensive Cash"

        elif regime == "crash":
            if shorting:
                return "Crisis Short"
            else:
                return "Capital Preservation"

        elif regime == "sideways":
            if shorting:
                return "Mean Reversion"
            else:
                return "Range Trading"

        else:
            return "Cautious Observer"

    def _get_strategy_description(
        self,
        regime: str,
        direction: str,
        action: str
    ) -> str:
        """Get strategy description."""
        descriptions = {
            ("strong_bull", "bullish", "buy"): "Strong uptrend detected. Actively seeking long entries.",
            ("strong_bull", "bullish", "hold"): "Strong uptrend but overbought. Holding positions.",
            ("bull", "bullish", "buy"): "Uptrend confirmed. Looking for pullback entries.",
            ("bull", "bullish", "hold"): "Uptrend intact. Managing existing positions.",
            ("strong_bear", "bearish", "sell"): "Strong downtrend. Actively shorting.",
            ("strong_bear", "bearish", "hold"): "Strong downtrend but oversold. Waiting.",
            ("bear", "bearish", "sell"): "Downtrend confirmed. Looking for short entries.",
            ("bear", "bearish", "hold"): "Bearish but cautious. Protecting capital.",
            ("crash", "bearish", "sell"): "Crisis mode. Maximum protection + opportunistic shorts.",
            ("crash", "bearish", "hold"): "Crisis mode. Preserving capital.",
            ("sideways", "neutral", "hold"): "Range-bound market. Waiting for breakout.",
        }

        key = (regime, direction, action)
        return descriptions.get(key, "Analyzing market conditions...")

    def get_current_strategy(self) -> Dict[str, Any]:
        """Get current strategy for API/dashboard."""
        return {
            "strategy": self.current_strategy.to_dict(),
            "risk_profile": self.current_profile.value,
            "settings": self.current_settings,
            "recent_decisions": [d.to_dict() for d in self.decision_history[-5:]],
        }

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """Get recent decision history."""
        return [d.to_dict() for d in self.decision_history[-limit:]]


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_controller: Optional[AdaptiveRiskController] = None

def get_adaptive_risk_controller() -> AdaptiveRiskController:
    """Get or create the global adaptive risk controller."""
    global _controller
    if _controller is None:
        _controller = AdaptiveRiskController()
    return _controller
