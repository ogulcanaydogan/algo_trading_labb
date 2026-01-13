"""
Reinforcement Learning Module for Trading.

Provides RL-based trading agents that learn optimal policies from
market interactions and trade outcomes.

Components:
- TradingEnvironment: Gym-compatible trading environment
- TradingPolicyNetwork: Neural network for policy learning
- PPOTrainer: Proximal Policy Optimization trainer
- HybridMLRLAgent: Combines ML predictions with RL policy
"""

from .environment import TradingEnvironment, TradingEnvConfig
from .policy_network import TradingPolicyNetwork, PolicyConfig
from .trainer import PPOTrainer, TrainingConfig, TrainingResults
from .hybrid_agent import HybridMLRLAgent, HybridConfig

__all__ = [
    # Environment
    "TradingEnvironment",
    "TradingEnvConfig",
    # Policy
    "TradingPolicyNetwork",
    "PolicyConfig",
    # Trainer
    "PPOTrainer",
    "TrainingConfig",
    "TrainingResults",
    # Hybrid
    "HybridMLRLAgent",
    "HybridConfig",
]
