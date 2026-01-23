"""
AI Orchestrator - Coordinates All AI Systems

The central coordinator that manages:
1. Parameter Optimizer - Finds optimal indicator parameters
2. Strategy Evolver - Discovers new strategies via genetic algorithms
3. RL Agent - Learns optimal actions through experience
4. Online Learner - Adapts in real-time
5. Meta-Allocator - Allocates capital across strategies
6. LLM Advisor - Provides context-aware reasoning

All systems work together to continuously improve trading performance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from .learning_db import LearningDatabase, get_learning_db
from .parameter_optimizer import ParameterOptimizer, OptimizationConfig, get_parameter_optimizer
from .strategy_evolver import StrategyEvolver, StrategyChromosome, get_strategy_evolver
from .rl_agent import RLTradingAgent, State, Action, get_rl_agent
from .online_learner import OnlineLearner, get_online_learner
from .meta_allocator import MetaAllocator, AllocationPlan, get_meta_allocator

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
AI_STATE_FILE = DATA_DIR / "ai_orchestrator_state.json"


class AIMode(Enum):
    """AI operation modes."""

    LEARNING = "learning"  # Actively learning from trades
    OPTIMIZING = "optimizing"  # Running parameter optimization
    EVOLVING = "evolving"  # Evolving new strategies
    TRADING = "trading"  # Normal trading with AI assistance
    PAPER = "paper"  # Paper trading for validation


@dataclass
class AIDecision:
    """
    Unified AI decision combining all AI systems.
    """

    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 - 1.0
    strategy_id: str
    parameters: Dict[str, Any]
    position_size_pct: float
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]

    # AI system contributions
    rl_action: str
    rl_confidence: float
    evolved_strategy_match: bool
    online_adjustment: float
    llm_override: Optional[str]

    # Reasoning
    reasoning: str
    warnings: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "strategy_id": self.strategy_id,
            "parameters": self.parameters,
            "position_size_pct": round(self.position_size_pct, 2),
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "rl_action": self.rl_action,
            "rl_confidence": round(self.rl_confidence, 3),
            "evolved_strategy_match": self.evolved_strategy_match,
            "online_adjustment": round(self.online_adjustment, 3),
            "llm_override": self.llm_override,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


class AIOrchestrator:
    """
    Master coordinator for all AI systems.

    Workflow:
    1. Receive market data and indicators
    2. Get RL agent action
    3. Check evolved strategies for matches
    4. Apply online learning adjustments
    5. Consult LLM for context (optional)
    6. Combine into final decision
    7. Track outcome for learning
    """

    def __init__(
        self,
        db: LearningDatabase = None,
        enable_rl: bool = True,
        enable_evolution: bool = True,
        enable_online_learning: bool = True,
        enable_llm: bool = True,
        llm_advisor=None,  # AITradingAdvisor instance
    ):
        self.db = db or get_learning_db()

        # Initialize AI systems
        self.parameter_optimizer = get_parameter_optimizer()
        self.strategy_evolver = get_strategy_evolver()
        self.rl_agent = get_rl_agent()
        self.online_learner = get_online_learner()
        self.meta_allocator = get_meta_allocator()

        # LLM advisor (optional, from ai_trading_advisor.py)
        self.llm_advisor = llm_advisor

        # Enable flags
        self.enable_rl = enable_rl
        self.enable_evolution = enable_evolution
        self.enable_online_learning = enable_online_learning
        self.enable_llm = enable_llm

        # State
        self.mode = AIMode.TRADING
        self.current_regime = "unknown"
        self.last_decision: Dict[str, AIDecision] = {}

        # Performance tracking
        self.decisions_made = 0
        self.successful_decisions = 0

        # Background tasks
        self._optimization_task: Optional[asyncio.Task] = None
        self._evolution_task: Optional[asyncio.Task] = None

        # Load state
        self._load_state()

    def _load_state(self):
        """Load orchestrator state from disk."""
        if AI_STATE_FILE.exists():
            try:
                with open(AI_STATE_FILE, "r") as f:
                    state = json.load(f)
                    self.decisions_made = state.get("decisions_made", 0)
                    self.successful_decisions = state.get("successful_decisions", 0)
                    self.current_regime = state.get("current_regime", "unknown")
                    logger.info(f"Loaded AI state: {self.decisions_made} decisions made")
            except Exception as e:
                logger.warning(f"Failed to load AI state: {e}")

    def _save_state(self):
        """Save orchestrator state to disk."""
        try:
            AI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "decisions_made": self.decisions_made,
                "successful_decisions": self.successful_decisions,
                "current_regime": self.current_regime,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(AI_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save AI state: {e}")

    async def get_decision(
        self,
        symbol: str,
        indicators: Dict[str, float],
        current_price: float,
        regime: str,
        position: float = 0,  # -1 short, 0 flat, 1 long
        position_pnl: float = 0,
        position_duration: int = 0,
        portfolio_value: float = 10000,
        base_signal: str = "FLAT",  # Technical signal
        base_confidence: float = 0.5,
    ) -> AIDecision:
        """
        Get unified AI decision for a trading opportunity.

        Args:
            symbol: Trading symbol
            indicators: Dict of indicator values
            current_price: Current market price
            regime: Market regime (bull, bear, sideways, etc.)
            position: Current position (-1, 0, 1)
            position_pnl: Current position P&L %
            position_duration: Bars in position
            portfolio_value: Total portfolio value
            base_signal: Technical signal from strategy
            base_confidence: Technical signal confidence

        Returns:
            AIDecision with combined AI recommendation
        """
        self.current_regime = regime
        warnings = []
        reasoning_parts = []

        # 1. Get RL agent action
        rl_action = "HOLD"
        rl_confidence = 0.5
        rl_probs = {"hold": 0.34, "buy": 0.33, "sell": 0.33}

        if self.enable_rl:
            try:
                state = State.from_market_data(
                    indicators, position, position_pnl, position_duration
                )
                action_idx = self.rl_agent.select_action(state, training=False)
                rl_action = ["HOLD", "BUY", "SELL"][action_idx]
                rl_probs = self.rl_agent.get_action_probabilities(state)
                rl_confidence = max(rl_probs.values())
                reasoning_parts.append(f"RL: {rl_action} ({rl_confidence:.0%})")
            except Exception as e:
                logger.warning(f"RL agent error: {e}")
                warnings.append("RL agent unavailable")

        # 2. Check evolved strategies
        evolved_match = False
        evolved_strategy = None

        if self.enable_evolution:
            try:
                top_strategies = self.strategy_evolver.get_best_strategies(3)
                for strategy in top_strategies:
                    if self._strategy_matches(strategy, indicators, regime):
                        evolved_match = True
                        evolved_strategy = strategy
                        reasoning_parts.append(f"Evolved strategy match: {strategy.id}")
                        break
            except Exception as e:
                logger.warning(f"Strategy evolver error: {e}")

        # 3. Apply online learning adjustments
        online_adjustment = 0.0
        should_trade = True

        if self.enable_online_learning:
            try:
                should_trade, adjustment, reason = self.online_learner.should_trade(
                    strategy_id=f"base_{symbol}",
                    regime=regime,
                    indicators=indicators,
                )
                online_adjustment = adjustment

                if not should_trade:
                    warnings.append(f"Online learner: {reason}")
                    reasoning_parts.append(f"Online: blocked ({reason})")
                elif adjustment != 0:
                    reasoning_parts.append(f"Online: adj {adjustment:+.0%}")
            except Exception as e:
                logger.warning(f"Online learner error: {e}")

        # 4. Get LLM advice (optional)
        llm_override = None

        if self.enable_llm and self.llm_advisor:
            try:
                advice = await self.llm_advisor.get_advice(
                    symbol=symbol,
                    current_price=current_price,
                    price_change_1h=indicators.get("price_change_1h", 0),
                    price_change_24h=indicators.get("price_change_24h", 0),
                    current_signal=base_signal,
                    regime=regime,
                    confidence=base_confidence,
                    portfolio_value=portfolio_value,
                    position_value=abs(position) * current_price,
                    pnl_pct=position_pnl,
                )

                if advice.action != base_signal and advice.confidence > 0.6:
                    llm_override = advice.action
                    reasoning_parts.append(f"LLM: {advice.action} - {advice.reasoning[:50]}")

                if advice.warnings:
                    warnings.extend(advice.warnings)

            except Exception as e:
                logger.warning(f"LLM advisor error: {e}")

        # 5. Combine signals into final decision
        final_action, final_confidence = self._combine_signals(
            base_signal=base_signal,
            base_confidence=base_confidence,
            rl_action=rl_action,
            rl_probs=rl_probs,
            evolved_match=evolved_match,
            evolved_strategy=evolved_strategy,
            online_adjustment=online_adjustment,
            should_trade=should_trade,
            llm_override=llm_override,
        )

        # 6. Get parameters
        params = self._get_best_parameters(symbol, regime)

        # 7. Calculate position size
        position_size = self._calculate_position_size(
            final_confidence,
            regime,
            portfolio_value,
        )

        # 8. Create decision
        decision = AIDecision(
            action=final_action,
            confidence=final_confidence,
            strategy_id=evolved_strategy.id if evolved_strategy else f"combined_{symbol}",
            parameters=params,
            position_size_pct=position_size,
            stop_loss_pct=params.get("atr_multiplier_sl", 2.0) * indicators.get("atr_pct", 1.0),
            take_profit_pct=params.get("atr_multiplier_tp", 3.0) * indicators.get("atr_pct", 1.0),
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            evolved_strategy_match=evolved_match,
            online_adjustment=online_adjustment,
            llm_override=llm_override,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Combined analysis",
            warnings=warnings,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.last_decision[symbol] = decision
        self.decisions_made += 1
        self._save_state()

        logger.info(
            f"AI Decision for {symbol}: {final_action} "
            f"(conf={final_confidence:.0%}, size={position_size:.1f}%)"
        )

        return decision

    def _combine_signals(
        self,
        base_signal: str,
        base_confidence: float,
        rl_action: str,
        rl_probs: Dict[str, float],
        evolved_match: bool,
        evolved_strategy: Optional[StrategyChromosome],
        online_adjustment: float,
        should_trade: bool,
        llm_override: Optional[str],
    ) -> Tuple[str, float]:
        """
        Combine signals from all AI systems into final decision.

        Weighting:
        - Base technical: 30%
        - RL agent: 25%
        - Evolved strategy: 20%
        - Online adjustment: 15%
        - LLM override: 10% (can override to HOLD)
        """
        if not should_trade:
            return "HOLD", 0.3

        # Calculate weighted scores for each action
        scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

        # Base signal contribution (30%)
        if base_signal in scores:
            scores[base_signal] += 0.30 * base_confidence
        else:
            scores["HOLD"] += 0.30 * 0.5

        # RL contribution (25%)
        scores["HOLD"] += 0.25 * rl_probs.get("hold", 0.34)
        scores["BUY"] += 0.25 * rl_probs.get("buy", 0.33)
        scores["SELL"] += 0.25 * rl_probs.get("sell", 0.33)

        # Evolved strategy contribution (20%)
        if evolved_match and evolved_strategy:
            # Determine evolved strategy signal based on genes
            evolved_signal = self._interpret_evolved_strategy(evolved_strategy)
            scores[evolved_signal] += 0.20 * min(1.0, evolved_strategy.fitness / 2.0)
        else:
            scores["HOLD"] += 0.20 * 0.5

        # Online adjustment (15%)
        # Positive adjustment boosts confidence in base signal
        if online_adjustment > 0 and base_signal in ["BUY", "SELL"]:
            scores[base_signal] += 0.15 * online_adjustment
        elif online_adjustment < 0:
            scores["HOLD"] += 0.15 * abs(online_adjustment)
        else:
            scores["HOLD"] += 0.15 * 0.3

        # LLM override (10%)
        if llm_override:
            if llm_override == "HOLD":
                scores["HOLD"] += 0.10 * 0.8
            elif llm_override in scores:
                scores[llm_override] += 0.10 * 0.6
        else:
            # No override, slight boost to base
            if base_signal in scores:
                scores[base_signal] += 0.10 * 0.3

        # Determine final action
        final_action = max(scores, key=scores.get)
        final_confidence = scores[final_action]

        # Normalize confidence to 0-1 range
        total_score = sum(scores.values())
        if total_score > 0:
            final_confidence = scores[final_action] / total_score

        # Apply minimum confidence threshold
        if final_action in ["BUY", "SELL"] and final_confidence < 0.4:
            final_action = "HOLD"
            final_confidence = 0.4

        return final_action, min(1.0, final_confidence)

    def _interpret_evolved_strategy(
        self,
        strategy: StrategyChromosome,
    ) -> str:
        """Interpret evolved strategy's signal direction."""
        # Check entry genes
        long_score = sum(g.weight for g in strategy.entry_long_genes)
        short_score = sum(g.weight for g in strategy.entry_short_genes)

        if long_score > short_score * 1.2:
            return "BUY"
        elif short_score > long_score * 1.2:
            return "SELL"
        return "HOLD"

    def _strategy_matches(
        self,
        strategy: StrategyChromosome,
        indicators: Dict[str, float],
        regime: str,
    ) -> bool:
        """Check if current conditions match an evolved strategy."""
        # Check regime preference
        if strategy.regime_preference and strategy.regime_preference != "all":
            if regime not in strategy.regime_preference:
                return False

        # Check entry gene conditions
        matches = 0
        total = len(strategy.entry_long_genes) + len(strategy.entry_short_genes)

        if total == 0:
            return False

        for gene in strategy.entry_long_genes + strategy.entry_short_genes:
            indicator_value = indicators.get(gene.indicator, None)
            if indicator_value is None:
                continue

            if self._gene_matches(gene, indicator_value):
                matches += 1

        # Require at least 60% of genes to match
        return matches / total >= 0.6

    def _gene_matches(
        self,
        gene,
        indicator_value: float,
    ) -> bool:
        """Check if a gene's condition is satisfied."""
        if gene.operator == ">":
            return indicator_value > gene.value
        elif gene.operator == "<":
            return indicator_value < gene.value
        elif gene.operator == "between":
            return gene.value <= indicator_value <= (gene.value2 or gene.value)
        return False

    def _get_best_parameters(
        self,
        symbol: str,
        regime: str,
    ) -> Dict[str, Any]:
        """Get best parameters for symbol/regime."""
        # Try to get optimized parameters
        params = self.parameter_optimizer.get_best_params(symbol, regime)

        if params:
            return params

        # Fall back to regime-adjusted defaults
        default_params = self.parameter_optimizer.get_default_params()
        return self.parameter_optimizer.get_regime_adjusted_params(default_params, regime)

    def _calculate_position_size(
        self,
        confidence: float,
        regime: str,
        portfolio_value: float,
    ) -> float:
        """Calculate position size based on confidence and regime."""
        # Base size from confidence
        base_size = confidence * 20  # Max 20% at 100% confidence

        # Regime adjustment
        regime_multipliers = {
            "strong_bull": 1.2,
            "bull": 1.0,
            "sideways": 0.7,
            "bear": 0.8,
            "strong_bear": 0.6,
            "volatile": 0.5,
            "crash": 0.3,
        }
        multiplier = regime_multipliers.get(regime, 0.8)

        final_size = base_size * multiplier

        # Cap at reasonable limits
        return max(2.0, min(25.0, final_size))

    async def on_trade_complete(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        indicators_at_entry: Dict[str, float],
        regime: str,
        hold_duration_mins: int,
    ):
        """
        Called when a trade completes. Updates all learning systems.
        """
        strategy_id = f"combined_{symbol}"

        # Update online learner
        if self.enable_online_learning:
            self.online_learner.on_trade_complete(
                symbol=symbol,
                strategy_id=strategy_id,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                regime=regime,
                indicators_at_entry=indicators_at_entry,
                hold_duration_mins=hold_duration_mins,
            )

        # Update RL agent
        if self.enable_rl:
            try:
                state = State.from_market_data(indicators_at_entry, position=1)
                next_state = State.from_market_data(indicators_at_entry, position=0)

                action = Action.BUY if exit_price > entry_price else Action.SELL
                reward = self.rl_agent.calculate_reward(
                    action=action,
                    pnl_pct=pnl_pct,
                    position_before=1,
                    position_after=0,
                )

                self.rl_agent.remember(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=True,
                    symbol=symbol,
                )

                # Train step
                self.rl_agent.train_step()
            except Exception as e:
                logger.warning(f"RL learning error: {e}")

        # Update meta-allocator
        last_decision = self.last_decision.get(symbol)
        if last_decision:
            self.meta_allocator.register_strategy(
                strategy_id=last_decision.strategy_id,
                metrics={
                    "sharpe_ratio": pnl_pct / 10,  # Rough estimate
                    "win_rate": 1.0 if pnl_pct > 0 else 0.0,
                    "trades": 1,
                },
            )

        # Track success
        if pnl_pct > 0:
            self.successful_decisions += 1

        self._save_state()

        logger.info(f"Trade complete recorded: {symbol} P&L={pnl_pct:+.2f}%")

    async def run_optimization(
        self,
        symbol: str,
        backtest_fn: Callable,
        regime: str = "all",
        n_trials: int = 50,
    ):
        """Run parameter optimization in background."""
        self.mode = AIMode.OPTIMIZING

        config = OptimizationConfig(
            symbol=symbol,
            regime=regime,
            n_trials=n_trials,
        )

        def progress(trial, total, best):
            logger.info(f"Optimization: {trial}/{total}, best={best:.4f}")

        best_params, best_score = await self.parameter_optimizer.optimize(
            config=config,
            backtest_fn=backtest_fn,
            progress_callback=progress,
        )

        self.mode = AIMode.TRADING

        logger.info(f"Optimization complete: best score={best_score:.4f}")
        return best_params, best_score

    async def run_evolution(
        self,
        fitness_fn: Callable,
        n_generations: int = 20,
    ):
        """Run strategy evolution in background."""
        self.mode = AIMode.EVOLVING

        def progress(gen, total, best):
            logger.info(f"Evolution: gen {gen}/{total}, best fitness={best:.4f}")

        best_strategy = await self.strategy_evolver.run_evolution(
            n_generations=n_generations,
            fitness_fn=fitness_fn,
            progress_callback=progress,
        )

        self.mode = AIMode.TRADING

        if best_strategy:
            logger.info(f"Evolution complete: best strategy fitness={best_strategy.fitness:.4f}")

        return best_strategy

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "mode": self.mode.value,
            "current_regime": self.current_regime,
            "decisions_made": self.decisions_made,
            "successful_decisions": self.successful_decisions,
            "success_rate": (
                self.successful_decisions / self.decisions_made if self.decisions_made > 0 else 0
            ),
            "systems": {
                "rl_enabled": self.enable_rl,
                "evolution_enabled": self.enable_evolution,
                "online_learning_enabled": self.enable_online_learning,
                "llm_enabled": self.enable_llm,
            },
            "rl_stats": self.rl_agent.get_stats() if self.enable_rl else {},
            "online_learning_status": (
                self.online_learner.get_learning_status() if self.enable_online_learning else {}
            ),
            "meta_allocation": self.meta_allocator.get_allocation_status(),
            "learning_db": self.db.get_learning_summary(),
        }


# Global instance
_orchestrator: Optional[AIOrchestrator] = None


def get_ai_orchestrator() -> AIOrchestrator:
    """Get or create global AI orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AIOrchestrator()
    return _orchestrator


async def initialize_ai_orchestrator(
    llm_advisor=None,
    enable_rl: bool = True,
    enable_evolution: bool = True,
    enable_online_learning: bool = True,
    enable_llm: bool = True,
) -> AIOrchestrator:
    """Initialize AI orchestrator with configuration."""
    global _orchestrator
    _orchestrator = AIOrchestrator(
        llm_advisor=llm_advisor,
        enable_rl=enable_rl,
        enable_evolution=enable_evolution,
        enable_online_learning=enable_online_learning,
        enable_llm=enable_llm,
    )
    return _orchestrator
