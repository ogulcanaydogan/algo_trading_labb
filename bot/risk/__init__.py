"""
Risk management module.

Provides:
- Drawdown recovery management
- Cross-market correlation management
- Portfolio Value-at-Risk (VaR)
"""

from .drawdown_recovery import (
    DrawdownRecoveryManager,
    DrawdownState,
    RecoveryPhase,
    RecoveryConfig,
    DailyPerformance,
    create_drawdown_manager,
)

from .correlation_manager import (
    CorrelationManager,
    CorrelationPair,
    CorrelationCluster,
    CorrelationConfig,
    ExposureCheck,
    create_correlation_manager,
)

from .portfolio_var import (
    PortfolioVaR,
    VaRResult,
    VaRConfig,
    PortfolioRiskMetrics,
    create_portfolio_var,
)

__all__ = [
    # Drawdown recovery
    "DrawdownRecoveryManager",
    "DrawdownState",
    "RecoveryPhase",
    "RecoveryConfig",
    "DailyPerformance",
    "create_drawdown_manager",
    # Correlation management
    "CorrelationManager",
    "CorrelationPair",
    "CorrelationCluster",
    "CorrelationConfig",
    "ExposureCheck",
    "create_correlation_manager",
    # Portfolio VaR
    "PortfolioVaR",
    "VaRResult",
    "VaRConfig",
    "PortfolioRiskMetrics",
    "create_portfolio_var",
]
