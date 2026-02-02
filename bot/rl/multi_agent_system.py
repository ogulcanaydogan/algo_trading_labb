"""
Multi-Agent RL System.

Specialized agents for different market conditions:
- TrendFollower: Bull markets, momentum plays
- MeanReversion: Sideways markets, range trading
- MomentumTrader: Breakouts, high volatility
- ShortSpecialist: Bear/crash markets
- MetaAgent: Selects which agent to use based on regime

Each agent has its own policy network trained on regime-specific data.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized trading agents."""
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM_TRADER = "momentum_trader"
    SHORT_SPECIALIST = "short_specialist"
    SCALPER = "scalper"


@dataclass
class AgentAction:
    """Action output from an agent."""
    action: str  # BUY, SELL, SHORT, COVER, HOLD
    confidence: float  # 0-1
    position_size_pct: float  # Suggested position size as % of portfolio
    stop_loss_pct: float  # Suggested stop loss
    take_profit_pct: float  # Suggested take profit
    leverage: float  # Suggested leverage
    reasoning: str  # Why this action
    agent_type: AgentType
    regime_match: float  # How well current regime matches agent specialty


@dataclass
class MarketState:
    """Current market state for agent decision making."""
    symbol: str
    price: float
    returns_1h: float = 0.0
    returns_4h: float = 0.0
    returns_24h: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0  # -1 to 1
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_position: float = 0.5  # 0-1
    volume_ratio: float = 1.0
    regime: str = "unknown"
    fear_greed: float = 50.0
    news_sentiment: float = 0.0


class BaseAgent(ABC):
    """Base class for specialized trading agents."""

    def __init__(
        self,
        agent_type: AgentType,
        model_path: Optional[str] = None,
    ):
        self.agent_type = agent_type
        self.model_path = model_path
        self._model = None
        self._performance_history: List[Dict] = []
        self._trade_count = 0
        self._win_count = 0

        # Regime affinity - how well this agent performs in each regime
        self.regime_affinity: Dict[str, float] = {}

    @abstractmethod
    def get_action(self, state: MarketState) -> AgentAction:
        """Get trading action for current state."""
        pass

    @abstractmethod
    def get_regime_match(self, regime: str) -> float:
        """Get how well current regime matches agent specialty (0-1)."""
        pass

    def record_outcome(self, pnl: float, pnl_pct: float, regime: str):
        """Record trade outcome for learning."""
        self._trade_count += 1
        if pnl > 0:
            self._win_count += 1

        # Update regime affinity
        if regime not in self.regime_affinity:
            self.regime_affinity[regime] = 0.5

        # Exponential moving average update
        alpha = 0.1
        outcome = 1.0 if pnl > 0 else 0.0
        self.regime_affinity[regime] = (
            alpha * outcome + (1 - alpha) * self.regime_affinity[regime]
        )

        self._performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "regime": regime,
        })

    @property
    def win_rate(self) -> float:
        return self._win_count / self._trade_count if self._trade_count > 0 else 0.5

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "win_rate": self.win_rate,
            "regime_affinity": self.regime_affinity,
        }


class TrendFollowerAgent(BaseAgent):
    """Agent specialized for trending markets."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(AgentType.TREND_FOLLOWER, model_path)
        # Strong affinity for bull/bear trends
        self.regime_affinity = {
            "BULL": 0.8,
            "STRONG_BULL": 0.9,
            "BEAR": 0.7,
            "STRONG_BEAR": 0.75,
            "SIDEWAYS": 0.3,
            "HIGH_VOL": 0.5,
            "LOW_VOL": 0.4,
        }

    def get_action(self, state: MarketState) -> AgentAction:
        """Follow the trend - buy strength, sell weakness."""
        action = "HOLD"
        confidence = 0.5
        reasoning = ""

        # Strong uptrend
        if state.trend_strength > 0.3 and state.rsi < 70:
            action = "BUY"
            confidence = min(0.9, 0.5 + state.trend_strength)
            reasoning = f"Strong uptrend (strength={state.trend_strength:.2f}), RSI not overbought"

        # Strong downtrend - short opportunity
        elif state.trend_strength < -0.3 and state.rsi > 30:
            action = "SHORT"
            confidence = min(0.9, 0.5 + abs(state.trend_strength))
            reasoning = f"Strong downtrend (strength={state.trend_strength:.2f}), RSI not oversold"

        # Trend weakening - exit
        elif abs(state.trend_strength) < 0.1:
            action = "HOLD"
            confidence = 0.6
            reasoning = "Trend weakening, staying flat"

        # Position sizing based on trend strength
        position_size = 0.1 + (abs(state.trend_strength) * 0.15)
        leverage = 1.0 + (abs(state.trend_strength) * 2.0)  # Up to 3x in strong trends

        return AgentAction(
            action=action,
            confidence=confidence,
            position_size_pct=min(0.25, position_size),
            stop_loss_pct=0.02,  # 2% stop for trend following
            take_profit_pct=0.06,  # 6% target (3:1 R:R)
            leverage=min(3.0, leverage),
            reasoning=reasoning,
            agent_type=self.agent_type,
            regime_match=self.get_regime_match(state.regime),
        )

    def get_regime_match(self, regime: str) -> float:
        return self.regime_affinity.get(regime, 0.5)


class MeanReversionAgent(BaseAgent):
    """Agent specialized for range-bound/sideways markets."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(AgentType.MEAN_REVERSION, model_path)
        self.regime_affinity = {
            "SIDEWAYS": 0.9,
            "LOW_VOL": 0.8,
            "BULL": 0.4,
            "BEAR": 0.4,
            "HIGH_VOL": 0.3,
            "CRASH": 0.1,
        }

    def get_action(self, state: MarketState) -> AgentAction:
        """Buy oversold, sell overbought."""
        action = "HOLD"
        confidence = 0.5
        reasoning = ""

        # Oversold - buy
        if state.rsi < 30 and state.bb_position < 0.2:
            action = "BUY"
            confidence = 0.7 + (30 - state.rsi) / 100
            reasoning = f"Oversold: RSI={state.rsi:.1f}, BB position={state.bb_position:.2f}"

        # Overbought - short
        elif state.rsi > 70 and state.bb_position > 0.8:
            action = "SHORT"
            confidence = 0.7 + (state.rsi - 70) / 100
            reasoning = f"Overbought: RSI={state.rsi:.1f}, BB position={state.bb_position:.2f}"

        # Near mean - close positions
        elif 0.4 < state.bb_position < 0.6:
            action = "HOLD"
            confidence = 0.6
            reasoning = "Price near mean, waiting for extremes"

        return AgentAction(
            action=action,
            confidence=min(0.9, confidence),
            position_size_pct=0.15,  # Conservative sizing
            stop_loss_pct=0.015,  # Tight stop for mean reversion
            take_profit_pct=0.025,  # Quick profits
            leverage=1.5,  # Lower leverage
            reasoning=reasoning,
            agent_type=self.agent_type,
            regime_match=self.get_regime_match(state.regime),
        )

    def get_regime_match(self, regime: str) -> float:
        return self.regime_affinity.get(regime, 0.5)


class MomentumTraderAgent(BaseAgent):
    """Agent specialized for breakouts and high momentum moves."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(AgentType.MOMENTUM_TRADER, model_path)
        self.regime_affinity = {
            "HIGH_VOL": 0.85,
            "STRONG_BULL": 0.8,
            "STRONG_BEAR": 0.75,
            "BREAKOUT": 0.9,
            "SIDEWAYS": 0.2,
            "LOW_VOL": 0.3,
        }

    def get_action(self, state: MarketState) -> AgentAction:
        """Catch momentum moves and breakouts."""
        action = "HOLD"
        confidence = 0.5
        reasoning = ""

        # Strong momentum up with volume
        if state.returns_1h > 0.02 and state.volume_ratio > 1.5:
            action = "BUY"
            confidence = 0.7 + min(0.2, state.returns_1h * 5)
            reasoning = f"Bullish momentum: +{state.returns_1h*100:.1f}% 1h, volume {state.volume_ratio:.1f}x"

        # Strong momentum down with volume - short
        elif state.returns_1h < -0.02 and state.volume_ratio > 1.5:
            action = "SHORT"
            confidence = 0.7 + min(0.2, abs(state.returns_1h) * 5)
            reasoning = f"Bearish momentum: {state.returns_1h*100:.1f}% 1h, volume {state.volume_ratio:.1f}x"

        # MACD crossover
        elif state.macd > state.macd_signal and state.macd > 0:
            action = "BUY"
            confidence = 0.65
            reasoning = "Bullish MACD crossover"

        elif state.macd < state.macd_signal and state.macd < 0:
            action = "SHORT"
            confidence = 0.65
            reasoning = "Bearish MACD crossover"

        return AgentAction(
            action=action,
            confidence=min(0.9, confidence),
            position_size_pct=0.12,
            stop_loss_pct=0.025,  # Wider stop for momentum
            take_profit_pct=0.05,  # Let winners run
            leverage=2.0,
            reasoning=reasoning,
            agent_type=self.agent_type,
            regime_match=self.get_regime_match(state.regime),
        )

    def get_regime_match(self, regime: str) -> float:
        return self.regime_affinity.get(regime, 0.5)


class ShortSpecialistAgent(BaseAgent):
    """Agent specialized for bear markets and shorting."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(AgentType.SHORT_SPECIALIST, model_path)
        self.regime_affinity = {
            "BEAR": 0.85,
            "STRONG_BEAR": 0.9,
            "CRASH": 0.95,
            "HIGH_VOL": 0.7,
            "BULL": 0.2,
            "SIDEWAYS": 0.4,
        }

    def get_action(self, state: MarketState) -> AgentAction:
        """Specialize in shorting during downtrends."""
        action = "HOLD"
        confidence = 0.5
        reasoning = ""

        # Strong bear conditions
        if state.trend_strength < -0.2 and state.returns_24h < -0.03:
            # Wait for bounce to short
            if state.rsi > 40:  # Some recovery
                action = "SHORT"
                confidence = 0.75 + abs(state.trend_strength) * 0.2
                reasoning = f"Bear market bounce short: RSI={state.rsi:.1f}, 24h={state.returns_24h*100:.1f}%"

        # Crash conditions - aggressive short
        elif state.returns_1h < -0.05 and state.volume_ratio > 2.0:
            action = "SHORT"
            confidence = 0.85
            reasoning = f"Crash detected: {state.returns_1h*100:.1f}% 1h drop with {state.volume_ratio:.1f}x volume"

        # Fear extreme - contrarian cover
        elif state.fear_greed < 20 and state.rsi < 25:
            action = "HOLD"  # Cover shorts, don't go long yet
            confidence = 0.6
            reasoning = "Extreme fear - covering shorts, waiting for reversal"

        # Position sizing - larger in confirmed bear
        position_size = 0.15 if state.regime in ["BEAR", "STRONG_BEAR", "CRASH"] else 0.08

        return AgentAction(
            action=action,
            confidence=min(0.9, confidence),
            position_size_pct=position_size,
            stop_loss_pct=0.03,  # 3% stop for shorts
            take_profit_pct=0.08,  # 8% target in crashes
            leverage=2.5,  # Higher leverage in bear
            reasoning=reasoning,
            agent_type=self.agent_type,
            regime_match=self.get_regime_match(state.regime),
        )

    def get_regime_match(self, regime: str) -> float:
        return self.regime_affinity.get(regime, 0.3)


class ScalperAgent(BaseAgent):
    """Agent specialized for quick scalping trades."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(AgentType.SCALPER, model_path)
        self.regime_affinity = {
            "SIDEWAYS": 0.8,
            "LOW_VOL": 0.7,
            "HIGH_VOL": 0.6,
            "BULL": 0.6,
            "BEAR": 0.6,
        }

    def get_action(self, state: MarketState) -> AgentAction:
        """Quick scalping based on short-term signals."""
        action = "HOLD"
        confidence = 0.5
        reasoning = ""

        # Scalp long on oversold bounce
        if state.bb_position < 0.15 and state.rsi < 35:
            action = "BUY"
            confidence = 0.7
            reasoning = f"Scalp long: BB={state.bb_position:.2f}, RSI={state.rsi:.1f}"

        # Scalp short on overbought rejection
        elif state.bb_position > 0.85 and state.rsi > 65:
            action = "SHORT"
            confidence = 0.7
            reasoning = f"Scalp short: BB={state.bb_position:.2f}, RSI={state.rsi:.1f}"

        return AgentAction(
            action=action,
            confidence=confidence,
            position_size_pct=0.08,  # Small position
            stop_loss_pct=0.008,  # Tight 0.8% stop
            take_profit_pct=0.015,  # Quick 1.5% profit
            leverage=3.0,  # Higher leverage for scalps
            reasoning=reasoning,
            agent_type=self.agent_type,
            regime_match=self.get_regime_match(state.regime),
        )

    def get_regime_match(self, regime: str) -> float:
        return self.regime_affinity.get(regime, 0.5)


class MetaAgent:
    """
    Meta-agent that selects and combines signals from specialized agents.

    Uses regime detection and agent performance history to weight agent outputs.
    """

    def __init__(self, data_dir: str = "data/rl"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize specialized agents
        self.agents: Dict[AgentType, BaseAgent] = {
            AgentType.TREND_FOLLOWER: TrendFollowerAgent(),
            AgentType.MEAN_REVERSION: MeanReversionAgent(),
            AgentType.MOMENTUM_TRADER: MomentumTraderAgent(),
            AgentType.SHORT_SPECIALIST: ShortSpecialistAgent(),
            AgentType.SCALPER: ScalperAgent(),
        }

        # Agent weights (learned over time)
        self.agent_weights: Dict[AgentType, float] = {
            agent_type: 1.0 for agent_type in AgentType
        }

        # Performance tracking
        self._decisions: List[Dict] = []
        self._load_state()

        logger.info(f"MetaAgent initialized with {len(self.agents)} specialized agents")

    def get_combined_action(
        self,
        state: MarketState,
        top_k: int = 2,
    ) -> Tuple[AgentAction, List[AgentAction]]:
        """
        Get combined action from top-k agents for current regime.

        Args:
            state: Current market state
            top_k: Number of top agents to combine

        Returns:
            Tuple of (best_action, all_actions)
        """
        # Get actions from all agents
        all_actions: List[AgentAction] = []
        for agent in self.agents.values():
            action = agent.get_action(state)
            all_actions.append(action)

        # Score each action based on:
        # 1. Regime match
        # 2. Agent historical performance
        # 3. Confidence
        scored_actions = []
        for action in all_actions:
            score = (
                action.regime_match * 0.4 +
                self.agents[action.agent_type].win_rate * 0.3 +
                action.confidence * 0.3
            ) * self.agent_weights[action.agent_type]
            scored_actions.append((score, action))

        # Sort by score
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # Get top-k actions
        top_actions = [a for _, a in scored_actions[:top_k]]

        # Combine top actions (weighted average for continuous values)
        if len(top_actions) == 1:
            best_action = top_actions[0]
        else:
            # Vote on action
            action_votes = {}
            for action in top_actions:
                action_votes[action.action] = action_votes.get(action.action, 0) + 1

            best_action_type = max(action_votes, key=action_votes.get)

            # Find the action with highest confidence for the winning vote
            best_action = max(
                [a for a in top_actions if a.action == best_action_type],
                key=lambda x: x.confidence
            )

            # Average the position sizing from agreeing agents
            agreeing = [a for a in top_actions if a.action == best_action_type]
            best_action = AgentAction(
                action=best_action_type,
                confidence=np.mean([a.confidence for a in agreeing]),
                position_size_pct=np.mean([a.position_size_pct for a in agreeing]),
                stop_loss_pct=np.mean([a.stop_loss_pct for a in agreeing]),
                take_profit_pct=np.mean([a.take_profit_pct for a in agreeing]),
                leverage=np.mean([a.leverage for a in agreeing]),
                reasoning=" | ".join([a.reasoning for a in agreeing if a.reasoning]),
                agent_type=agreeing[0].agent_type,
                regime_match=np.mean([a.regime_match for a in agreeing]),
            )

        # Log decision
        self._decisions.append({
            "timestamp": datetime.now().isoformat(),
            "regime": state.regime,
            "action": best_action.action,
            "agent": best_action.agent_type.value,
            "confidence": best_action.confidence,
        })

        return best_action, all_actions

    def record_outcome(
        self,
        agent_type: AgentType,
        pnl: float,
        pnl_pct: float,
        regime: str,
    ):
        """Record trade outcome and update agent weights."""
        # Update the specific agent
        self.agents[agent_type].record_outcome(pnl, pnl_pct, regime)

        # Update agent weight based on outcome
        alpha = 0.05
        outcome_score = 1.0 if pnl > 0 else 0.5 if pnl == 0 else 0.0
        self.agent_weights[agent_type] = (
            alpha * outcome_score + (1 - alpha) * self.agent_weights[agent_type]
        )

        # Normalize weights
        total = sum(self.agent_weights.values())
        self.agent_weights = {k: v / total * len(self.agent_weights) for k, v in self.agent_weights.items()}

        self._save_state()

    def get_best_agent_for_regime(self, regime: str) -> AgentType:
        """Get the best performing agent for a given regime."""
        best_agent = None
        best_score = -1

        for agent_type, agent in self.agents.items():
            score = agent.get_regime_match(regime) * agent.win_rate * self.agent_weights[agent_type]
            if score > best_score:
                best_score = score
                best_agent = agent_type

        return best_agent or AgentType.TREND_FOLLOWER

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-agent statistics."""
        return {
            "agent_weights": {k.value: v for k, v in self.agent_weights.items()},
            "agent_stats": {k.value: v.get_stats() for k, v in self.agents.items()},
            "total_decisions": len(self._decisions),
        }

    def _save_state(self):
        """Save agent states to disk."""
        state = {
            "agent_weights": {k.value: v for k, v in self.agent_weights.items()},
            "agent_stats": {k.value: v.get_stats() for k, v in self.agents.items()},
        }
        with open(self.data_dir / "meta_agent_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load agent states from disk."""
        state_file = self.data_dir / "meta_agent_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.agent_weights = {
                    AgentType(k): v for k, v in state.get("agent_weights", {}).items()
                }
                logger.info("Loaded MetaAgent state from disk")
            except Exception as e:
                logger.warning(f"Failed to load MetaAgent state: {e}")


# Singleton instance
_meta_agent: Optional[MetaAgent] = None


def get_meta_agent() -> MetaAgent:
    """Get or create the MetaAgent singleton."""
    global _meta_agent
    if _meta_agent is None:
        _meta_agent = MetaAgent()
    return _meta_agent
