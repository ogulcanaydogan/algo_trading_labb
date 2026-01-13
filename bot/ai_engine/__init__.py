"""
Multi-AI Trading Engine

A comprehensive AI system that continuously improves trading strategies through:
1. Parameter Optimization - Finds optimal indicator parameters
2. Strategy Evolution - Genetic algorithm discovers new strategies
3. Reinforcement Learning - Agent learns optimal actions
4. Online Learning - Adapts in real-time to market changes
5. Meta-Allocation - Allocates capital across best strategies
6. LLM Reasoning - Provides context-aware decision making
7. Leverage Optimization - Learns optimal leverage and shorting strategies

All AI systems work together, coordinated by the AI Orchestrator.
"""

from .orchestrator import AIOrchestrator, get_ai_orchestrator
from .parameter_optimizer import ParameterOptimizer
from .strategy_evolver import StrategyEvolver
from .rl_agent import RLTradingAgent
from .online_learner import OnlineLearner
from .meta_allocator import MetaAllocator
from .learning_db import LearningDatabase
from .leverage_rl_agent import (
    LeverageRLAgent,
    LeverageState,
    LeverageAction,
    get_leverage_rl_agent,
)
from .leverage_manager import (
    AILeverageManager,
    LeverageDecision,
    MarginStatus,
    get_leverage_manager,
)

__all__ = [
    # Core orchestration
    'AIOrchestrator',
    'get_ai_orchestrator',
    # Optimization
    'ParameterOptimizer',
    'StrategyEvolver',
    # RL Agents
    'RLTradingAgent',
    'LeverageRLAgent',
    'LeverageState',
    'LeverageAction',
    'get_leverage_rl_agent',
    # Learning systems
    'OnlineLearner',
    'MetaAllocator',
    'LearningDatabase',
    # Leverage management
    'AILeverageManager',
    'LeverageDecision',
    'MarginStatus',
    'get_leverage_manager',
]
