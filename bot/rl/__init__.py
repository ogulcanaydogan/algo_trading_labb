"""
Reinforcement Learning Module for Trading.

Provides RL-based trading agents that learn optimal policies from
market interactions and trade outcomes.

Components:
- TradingEnvironment: Gym-compatible trading environment
- TradingPolicyNetwork: Neural network for policy learning
- PPOTrainer: Proximal Policy Optimization trainer
- A2CTrainer: Advantage Actor-Critic trainer
- HybridMLRLAgent: Combines ML predictions with RL policy
- RLPositionSizer: RL-based dynamic position sizing
"""

from .environment import TradingEnvironment, TradingEnvConfig
from .policy_network import TradingPolicyNetwork, PolicyConfig
from .trainer import PPOTrainer, TrainingConfig, TrainingResults
from .a2c_trainer import A2CTrainer, A2CConfig, A2CResults, MultiEnvA2CTrainer
from .hybrid_agent import HybridMLRLAgent, HybridConfig
from .position_sizer import (
    RLPositionSizer,
    DiscretePositionSizer,
    PositionSizerConfig,
    PositionSizerState,
)

__all__ = [
    # Environment
    "TradingEnvironment",
    "TradingEnvConfig",
    # Policy
    "TradingPolicyNetwork",
    "PolicyConfig",
    # PPO Trainer
    "PPOTrainer",
    "TrainingConfig",
    "TrainingResults",
    # A2C Trainer
    "A2CTrainer",
    "A2CConfig",
    "A2CResults",
    "MultiEnvA2CTrainer",
    # Hybrid
    "HybridMLRLAgent",
    "HybridConfig",
    # Position Sizing
    "RLPositionSizer",
    "DiscretePositionSizer",
    "PositionSizerConfig",
    "PositionSizerState",
]
