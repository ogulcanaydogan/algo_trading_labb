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

from .dynamic_controls import (
    CircuitBreakerState,
    RiskLevel,
    CorrelationAlert,
    CircuitBreakerStatus,
    CorrelationConfig,
    CorrelationCircuitBreaker,
    PositionSizingConfig as DynamicPositionSizingConfig,
    PositionSizeResult,
    DynamicPositionSizer,
    create_correlation_circuit_breaker,
    create_dynamic_position_sizer,
)

from .stress_testing import (
    ScenarioType,
    RiskFactor,
    StressScenario,
    StressResult,
    StressTestReport,
    Position as StressTestPosition,
    HistoricalScenarios,
    HypotheticalScenarios,
    StressTestEngine,
    create_stress_test_engine,
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
    # Dynamic Controls
    "CircuitBreakerState",
    "RiskLevel",
    "CorrelationAlert",
    "CircuitBreakerStatus",
    "CorrelationConfig",
    "CorrelationCircuitBreaker",
    "DynamicPositionSizingConfig",
    "PositionSizeResult",
    "DynamicPositionSizer",
    "create_correlation_circuit_breaker",
    "create_dynamic_position_sizer",
    # Stress Testing
    "ScenarioType",
    "RiskFactor",
    "StressScenario",
    "StressResult",
    "StressTestReport",
    "StressTestPosition",
    "HistoricalScenarios",
    "HypotheticalScenarios",
    "StressTestEngine",
    "create_stress_test_engine",
]
