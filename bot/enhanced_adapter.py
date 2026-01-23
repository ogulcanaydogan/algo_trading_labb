"""
Enhanced Adapter Module.

Bridges the EnhancedTradingEngine components with the existing
UnifiedTradingEngine infrastructure, providing:
- Advanced execution algorithms
- Regime-aware strategy selection
- Dynamic position sizing
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .execution_adapter import (
    ExecutionAdapter,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from .execution import (
    create_execution_algorithm,
    AlgoOrder,
    UrgencyLevel,
)
from .regime import (
    MarketRegime,
    RegimeDetector,
    create_regime_strategy_selector,
)
from .risk import (
    create_correlation_circuit_breaker,
    create_dynamic_position_sizer,
    create_stress_test_engine,
    CircuitBreakerState,
    StressTestPosition,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAdapterConfig:
    """Configuration for enhanced adapter."""

    # Execution settings
    execution_algorithm: str = "adaptive"  # twap, vwap, pov, is, iceberg, adaptive
    execution_urgency: UrgencyLevel = UrgencyLevel.MEDIUM

    # Risk controls
    enable_circuit_breaker: bool = True
    enable_stress_testing: bool = True
    max_stress_loss_pct: float = 0.25

    # Position sizing
    base_risk_per_trade: float = 0.02
    max_position_size: float = 0.20
    target_volatility: float = 0.15

    # Strategy selection
    enable_strategy_selection: bool = True
    max_concurrent_strategies: int = 3


class EnhancedExecutionAdapter:
    """
    Enhanced execution adapter that wraps the base ExecutionAdapter
    with advanced features from the enhanced trading modules.

    Usage:
        base_adapter = PaperExecutionAdapter(initial_balance=10000)
        enhanced = EnhancedExecutionAdapter(base_adapter, config)
        await enhanced.execute_signal(symbol, signal, ohlcv_data)
    """

    def __init__(
        self,
        base_adapter: ExecutionAdapter,
        config: Optional[EnhancedAdapterConfig] = None,
    ):
        self.base = base_adapter
        self.config = config or EnhancedAdapterConfig()

        # Initialize enhanced components
        self._init_components()

        # State
        self._portfolio_value: float = 0
        self._current_regime: Optional[MarketRegime] = None
        self._returns_buffer: Dict[str, List[float]] = {}

        logger.info("Enhanced execution adapter initialized")

    def _init_components(self):
        """Initialize enhanced trading components."""
        # Regime detector
        self.regime_detector = RegimeDetector()

        # Strategy selector
        if self.config.enable_strategy_selection:
            self.strategy_selector = create_regime_strategy_selector(include_defaults=True)
        else:
            self.strategy_selector = None

        # Circuit breaker
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = create_correlation_circuit_breaker()
        else:
            self.circuit_breaker = None

        # Position sizer
        self.position_sizer = create_dynamic_position_sizer(
            circuit_breaker=self.circuit_breaker
        )

        # Stress test engine
        if self.config.enable_stress_testing:
            self.stress_engine = create_stress_test_engine()
        else:
            self.stress_engine = None

    # ========================================
    # Pass-through methods to base adapter
    # ========================================

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        return await self.base.get_balance()

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return await self.base.get_positions()

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        return await self.base.get_current_price(symbol)

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data."""
        return await self.base.get_ohlcv(symbol, timeframe, limit)

    # ========================================
    # Enhanced methods
    # ========================================

    async def update_state(self, symbol: str, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Update internal state with new market data.

        Returns:
            Dict with current regime, circuit breaker status, etc.
        """
        # Update portfolio value
        balance = await self.get_balance()
        self._portfolio_value = balance.get("total", 0)
        self.position_sizer.set_portfolio_value(self._portfolio_value)

        # Detect regime
        regime_state = self.regime_detector.detect(ohlcv, symbol)
        self._current_regime = regime_state.regime

        # Update circuit breaker with returns
        if self.circuit_breaker and len(ohlcv) > 20:
            returns = ohlcv["close"].pct_change().dropna().tolist()
            if symbol not in self._returns_buffer:
                self._returns_buffer[symbol] = []
            self._returns_buffer[symbol] = returns[-100:]

            # Create returns DataFrame for correlation tracking
            if len(self._returns_buffer) > 1:
                returns_df = pd.DataFrame(self._returns_buffer)
                self.circuit_breaker.update_correlations(returns_df)

        return {
            "regime": self._current_regime.value if self._current_regime else "unknown",
            "regime_confidence": regime_state.confidence,
            "circuit_breaker_state": (
                self.circuit_breaker.state.value if self.circuit_breaker else "disabled"
            ),
            "portfolio_value": self._portfolio_value,
            "risk_level": self.position_sizer.get_portfolio_risk_level().value,
        }

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        current_volatility: float = 0.15,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using dynamic sizer.

        Returns:
            Dict with final_size and sizing details
        """
        regime = self._current_regime.value if self._current_regime else "unknown"

        result = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            current_volatility=current_volatility,
            regime=regime,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
        )

        return result.to_dict()

    def get_strategy_selection(self, volatility: float = 0.15) -> Dict[str, Any]:
        """
        Get strategy selection based on current regime.

        Returns:
            Dict with selected strategies and allocations
        """
        if not self.strategy_selector or not self._current_regime:
            return {"selected_strategies": [], "position_scale": 1.0}

        result = self.strategy_selector.select_strategies(
            self._current_regime,
            volatility=volatility,
            force_reselection=True,
        )

        return {
            "selected_strategies": result.selected_strategies,
            "regime": result.regime.value,
            "position_scale": result.position_scale,
            "risk_scale": result.risk_scale,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    def run_stress_test(
        self,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run stress test on proposed positions.

        Args:
            positions: List of position dicts with symbol, quantity, price, asset_class

        Returns:
            Stress test results
        """
        if not self.stress_engine:
            return {"enabled": False}

        stress_positions = [
            StressTestPosition(
                symbol=p["symbol"],
                quantity=p["quantity"],
                current_price=p["price"],
                asset_class=p.get("asset_class", "equity"),
                beta=p.get("beta", 1.0),
            )
            for p in positions
        ]

        self.stress_engine.set_portfolio(stress_positions)
        report = self.stress_engine.run_all_scenarios()

        return {
            "scenarios_tested": report.scenarios_tested,
            "worst_case_loss": report.worst_case.portfolio_pnl_pct if report.worst_case else 0,
            "expected_shortfall": report.expected_shortfall,
            "recommendations": report.recommendations,
            "is_acceptable": (
                report.worst_case.portfolio_pnl_pct > -self.config.max_stress_loss_pct
                if report.worst_case
                else True
            ),
        }

    async def execute_with_algorithm(
        self,
        order: Order,
        algorithm: Optional[str] = None,
    ) -> OrderResult:
        """
        Execute order using specified execution algorithm.

        Args:
            order: Order to execute
            algorithm: Algorithm name (twap, vwap, etc.) or None for default

        Returns:
            OrderResult with execution details
        """
        algo_name = algorithm or self.config.execution_algorithm

        # For simple market orders, use base adapter directly
        if algo_name == "market" or order.quantity < 0.01:
            return await self.base.execute_order(order)

        # Create execution algorithm
        async def order_executor(**kwargs):
            exec_order = Order(
                symbol=kwargs.get("symbol", order.symbol),
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=kwargs.get("quantity", order.quantity),
            )
            result = await self.base.execute_order(exec_order)
            return {
                "fill_price": result.average_price,
                "filled_quantity": result.filled_quantity,
            }

        async def market_data_provider(symbol: str):
            price = await self.get_current_price(symbol)
            return {
                "mid_price": price,
                "last_price": price,
                "bid_price": price * 0.9999,
                "ask_price": price * 1.0001,
                "bid_size": 1000,
                "ask_size": 1000,
                "volume": 10000,
                "volatility": 0.02,
                "spread_bps": 2,
            }

        try:
            algo = create_execution_algorithm(
                algo_name,
                order_executor,
                market_data_provider,
            )

            algo_order = AlgoOrder(
                order_id=order.client_order_id or f"order_{datetime.now().timestamp()}",
                symbol=order.symbol,
                side="buy" if order.side == OrderSide.BUY else "sell",
                total_quantity=order.quantity,
                urgency=self.config.execution_urgency,
            )

            execution = await algo.execute(algo_order)

            return OrderResult(
                success=execution.filled_quantity > 0,
                order_id=execution.order_id,
                status=OrderStatus.FILLED if execution.fill_rate >= 0.99 else OrderStatus.PARTIALLY_FILLED,
                filled_quantity=execution.filled_quantity,
                average_price=execution.average_price,
                commission=0,
                simulated=True,
                slippage_applied=execution.slippage_bps / 10000,
            )

        except Exception as e:
            logger.error(f"Algorithm execution failed: {e}, falling back to market order")
            return await self.base.execute_order(order)

    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        Check if trading is currently allowed based on risk controls.

        Returns:
            Tuple of (allowed, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker:
            if self.circuit_breaker.state == CircuitBreakerState.TRIGGERED:
                return False, "Circuit breaker triggered"

        # Check risk level
        risk_level = self.position_sizer.get_portfolio_risk_level()
        if risk_level.value == "critical":
            return False, "Portfolio risk level is critical"

        return True, "Trading allowed"

    def get_status(self) -> Dict[str, Any]:
        """Get current adapter status."""
        return {
            "base_adapter": type(self.base).__name__,
            "config": {
                "execution_algorithm": self.config.execution_algorithm,
                "circuit_breaker_enabled": self.config.enable_circuit_breaker,
                "stress_testing_enabled": self.config.enable_stress_testing,
            },
            "state": {
                "portfolio_value": self._portfolio_value,
                "current_regime": self._current_regime.value if self._current_regime else None,
                "circuit_breaker": (
                    self.circuit_breaker.get_status().to_dict()
                    if self.circuit_breaker
                    else None
                ),
                "risk_level": self.position_sizer.get_portfolio_risk_level().value,
            },
        }


def create_enhanced_adapter(
    base_adapter: ExecutionAdapter,
    config: Optional[EnhancedAdapterConfig] = None,
) -> EnhancedExecutionAdapter:
    """
    Factory function to create enhanced execution adapter.

    Args:
        base_adapter: Base execution adapter (paper, testnet, live)
        config: Optional configuration

    Returns:
        EnhancedExecutionAdapter instance
    """
    return EnhancedExecutionAdapter(base_adapter, config)
