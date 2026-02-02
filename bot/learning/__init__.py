"""
Learning Module - Self-Learning Infrastructure.

This module provides the learning infrastructure for the adaptive AI trading system:
- Unified Learning Database: Stores all trade experiences with full context
- Feedback Orchestrator: Coordinates learning updates across all systems
- Regime-aware Experience Buffer: Smart sampling based on market conditions
"""

from bot.learning.learning_database import (
    LearningDatabase,
    TradeRecord,
    LearningMetrics,
)
from bot.learning.feedback_orchestrator import (
    FeedbackOrchestrator,
    FeedbackConfig,
)

__all__ = [
    "LearningDatabase",
    "TradeRecord",
    "LearningMetrics",
    "FeedbackOrchestrator",
    "FeedbackConfig",
]
