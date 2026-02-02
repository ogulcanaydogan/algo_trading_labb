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
- MultiAgentSystem: Specialized agents for different market conditions
- RewardShaper: 1% daily target reward optimization
- PortfolioAgent: Capital allocation optimization
- ContinualLearning: Prevent catastrophic forgetting
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

# New Phase 2 components
from .multi_agent_system import (
    MetaAgent,
    AgentType,
    AgentAction,
    MarketState as RLMarketState,
    TrendFollowerAgent,
    MeanReversionAgent,
    MomentumTraderAgent,
    ShortSpecialistAgent,
    ScalperAgent,
    get_meta_agent,
)
from .reward_shaping import (
    RewardShaper,
    DailyProgress,
    get_reward_shaper,
)
from .portfolio_agent import (
    PortfolioAgent,
    AssetMetrics,
    PortfolioState,
    get_portfolio_agent,
)
from .continual_learning import (
    ContinualLearningManager,
    ElasticWeightConsolidation,
    ReservoirSampler,
    RegimeSpecificBuffer,
    get_continual_learning_manager,
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
    # Multi-Agent System (Phase 2)
    "MetaAgent",
    "AgentType",
    "AgentAction",
    "RLMarketState",
    "TrendFollowerAgent",
    "MeanReversionAgent",
    "MomentumTraderAgent",
    "ShortSpecialistAgent",
    "ScalperAgent",
    "get_meta_agent",
    # Reward Shaping (Phase 2)
    "RewardShaper",
    "DailyProgress",
    "get_reward_shaper",
    # Portfolio Agent (Phase 2)
    "PortfolioAgent",
    "AssetMetrics",
    "PortfolioState",
    "get_portfolio_agent",
    # Continual Learning (Phase 2)
    "ContinualLearningManager",
    "ElasticWeightConsolidation",
    "ReservoirSampler",
    "RegimeSpecificBuffer",
    "get_continual_learning_manager",
]
