"""
A/B Testing Framework for Strategy Comparison

Run multiple strategies in parallel with paper portfolios,
collect performance metrics, and perform statistical comparison.
"""

from bot.testing.ab_framework import (
    ABTest,
    ABTestConfig,
    ABTestResult,
    PaperPortfolio,
    StrategyRunner,
    StatisticalComparison,
    run_ab_test,
)

__all__ = [
    "ABTest",
    "ABTestConfig",
    "ABTestResult",
    "PaperPortfolio",
    "StrategyRunner",
    "StatisticalComparison",
    "run_ab_test",
]
