"""
Portfolio module for optimization and management.

Provides:
- Black-Litterman optimization
- Risk Parity allocation
- Mean-Variance optimization
- Minimum Volatility portfolios
"""

from .optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    OptimizationConfig,
    BlackLittermanView,
    create_portfolio_optimizer,
)

__all__ = [
    "PortfolioOptimizer",
    "OptimizationResult",
    "OptimizationConfig",
    "BlackLittermanView",
    "create_portfolio_optimizer",
]
