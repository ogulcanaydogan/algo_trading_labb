"""
Feature management module for feature flags and runtime configuration.

Provides:
- Feature flag management
- Gradual rollout support
- Targeting rules for feature control
"""

from .feature_flags import (
    FeatureFlagManager,
    TradingFeatureFlags,
    FeatureFlag,
    FlagType,
    RolloutStrategy,
    TargetingRule,
    EvaluationResult,
    create_feature_flag_manager,
    create_trading_flags,
)

__all__ = [
    "FeatureFlagManager",
    "TradingFeatureFlags",
    "FeatureFlag",
    "FlagType",
    "RolloutStrategy",
    "TargetingRule",
    "EvaluationResult",
    "create_feature_flag_manager",
    "create_trading_flags",
]
