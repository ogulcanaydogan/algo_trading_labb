"""
Testing module for A/B testing and experimentation.

Provides:
- A/B Testing Framework
- Strategy comparison tests
- Statistical analysis tools
"""

from .ab_testing import (
    ABTestingFramework,
    Experiment,
    ExperimentStatus,
    ExperimentResult,
    Variant,
    MetricResult,
    AllocationMethod,
    StrategyABTest,
    create_ab_testing_framework,
    create_strategy_ab_test,
)

__all__ = [
    "ABTestingFramework",
    "Experiment",
    "ExperimentStatus",
    "ExperimentResult",
    "Variant",
    "MetricResult",
    "AllocationMethod",
    "StrategyABTest",
    "create_ab_testing_framework",
    "create_strategy_ab_test",
]
