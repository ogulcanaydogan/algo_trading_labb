"""
Deployment Module.

Provides safe deployment capabilities:
- Gradual rollout from paper to live
- Stage-based capital allocation
- Automatic rollback triggers
"""

from .gradual_rollout import (
    GradualRolloutSystem,
    RollbackEvent,
    RolloutMetrics,
    RolloutStage,
    StageConfig,
    get_gradual_rollout,
)

__all__ = [
    "GradualRolloutSystem",
    "RollbackEvent",
    "RolloutMetrics",
    "RolloutStage",
    "StageConfig",
    "get_gradual_rollout",
]
