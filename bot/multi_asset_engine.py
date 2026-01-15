"""
Multi-Asset Trading Engine.

Integrates portfolio optimization with the smart trading engine
to manage multiple assets with optimal allocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .smart_engine import SmartTradingEngine, TradingDecision, EngineConfig
from .portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationMethod,
    AllocationResult,
    MultiAssetPortfolioManager,
)
from .risk_manager import RiskManager, RiskConfig, RiskAssessment
from .position_manager import PositionManager, Position, PositionConfig
from .database import TradingDatabase, Trade

# Import data store for persistent trade recording
try:
    from .data_store import get_data_store, record_trade as store_trade
    HAS_DATA_STORE = True
except ImportError:
    HAS_DATA_STORE = False

# Import order validation
try:
    from .order_validator import validate_order, OrderValidation
    HAS_ORDER_VALIDATOR = True
except ImportError:
    HAS_ORDER_VALIDATOR = False

# Import data freshness monitor
try:
    from .data_freshness import get_monitor as get_freshness_monitor, update_data
    HAS_DATA_FRESHNESS = True
except ImportError:
    HAS_DATA_FRESHNESS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-asset-engine")


@dataclass
class AssetConfig:
    """Configuration for a single asset."""
    symbol: str
    data_symbol: Optional[str] = None  # For data provider mapping
    asset_type: str = "crypto"
    min_weight: float = 0.0
    max_weight: float = 1.0
    enabled: bool = True


@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset trading."""
    assets: List[AssetConfig] = field(default_factory=list)
    optimization_method: OptimizationMethod = OptimizationMethod.RISK_PARITY
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalance
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    total_capital: float = 10000.0
    risk_free_rate: float = 0.02
    min_history_days: int = 30  # Minimum days of data needed
    use_correlation_filter: bool = True
    max_correlation: float = 0.85  # Skip highly correlated assets


@dataclass
class AssetDecision:
    """Trading decision for a single asset within the portfolio."""
    symbol: str
    action: str
    weight: float
    target_value: float
    current_value: float
    target_quantity: float
    confidence: float
    regime: str
    reasoning: List[str]


@dataclass
class PortfolioDecision:
    """Complete portfolio trading decision."""
    timestamp: datetime
    asset_decisions: List[AssetDecision]
    allocation: Dict[str, float]
    portfolio_metrics: Dict[str, float]
    rebalance_needed: bool
    trades_to_execute: List[Dict[str, Any]]
    total_value: float
    cash_balance: float

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_decisions": [
                {
                    "symbol": d.symbol,
                    "action": d.action,
                    "weight": round(d.weight, 4),
                    "target_value": round(d.target_value, 2),
                    "confidence": round(d.confidence, 4),
                    "regime": d.regime,
                }
                for d in self.asset_decisions
            ],
            "allocation": {k: round(v, 4) for k, v in self.allocation.items()},
            "portfolio_metrics": {k: round(v, 4) for k, v in self.portfolio_metrics.items()},
            "rebalance_needed": self.rebalance_needed,
            "trades_count": len(self.trades_to_execute),
            "total_value": round(self.total_value, 2),
            "cash_balance": round(self.cash_balance, 2),
        }

    def print_summary(self):
        """Print portfolio decision summary."""
        print("\n" + "=" * 70)
        print("PORTFOLIO DECISION")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Value: ${self.total_value:,.2f}")
        print(f"Cash Balance: ${self.cash_balance:,.2f}")
        print(f"Rebalance Needed: {'Yes' if self.rebalance_needed else 'No'}")

        print("\nAsset Allocations:")
        print("-" * 70)
        print(f"{'Symbol':<12} {'Action':<8} {'Weight':<10} {'Target $':<12} {'Confidence':<10} {'Regime':<12}")
        print("-" * 70)
        for d in self.asset_decisions:
            print(f"{d.symbol:<12} {d.action:<8} {d.weight:.2%}     ${d.target_value:<10,.2f} {d.confidence:.2%}     {d.regime:<12}")

        if self.trades_to_execute:
            print("\nTrades to Execute:")
            print("-" * 70)
            for trade in self.trades_to_execute:
                print(f"  {trade['action']} {trade['quantity']:.6f} {trade['symbol']} @ ~${trade['price']:,.2f}")

        print("\nPortfolio Metrics:")
        for metric, value in self.portfolio_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("=" * 70)


class MultiAssetTradingEngine:
    """
    Multi-Asset Trading Engine.

    Manages a portfolio of assets with:
    1. Individual smart trading analysis per asset
    2. Portfolio-level optimization
    3. Correlation-aware allocation
    4. Risk parity or other optimization methods
    5. Automatic rebalancing
    """

    def __init__(
        self,
        config: Optional[MultiAssetConfig] = None,
        engine_config: Optional[EngineConfig] = None,
        data_dir: str = "data",
    ):
        self.config = config or MultiAssetConfig()
        self.engine_config = engine_config or EngineConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize portfolio optimizer
        self.portfolio_optimizer = PortfolioOptimizer(
            risk_free_rate=self.config.risk_free_rate,
        )

        self.portfolio_manager = MultiAssetPortfolioManager(
            optimizer=self.portfolio_optimizer,
            rebalance_threshold=self.config.rebalance_threshold,
            rebalance_frequency=self.config.rebalance_frequency,
        )

        # Initialize per-asset smart engines
        self.asset_engines: Dict[str, SmartTradingEngine] = {}
        for asset in self.config.assets:
            if asset.enabled:
                self.asset_engines[asset.symbol] = SmartTradingEngine(
                    config=self.engine_config,
                    data_dir=str(self.data_dir / "engines" / asset.symbol.replace("/", "_")),
                )

        # Risk manager for portfolio-level risk
        self.risk_manager = RiskManager(RiskConfig(
            max_daily_loss_pct=5.0,  # 5% max daily loss
            max_drawdown_pct=15.0,   # 15% max drawdown
            max_single_position_pct=self.config.assets[0].max_weight * 100 if self.config.assets else 50.0,
        ))

        # Position manager for tracking
        self.position_manager = PositionManager(PositionConfig())

        # Database for persistence
        self.database = TradingDatabase(str(self.data_dir / "portfolio.db"))

        # State
        self._current_allocation: Optional[AllocationResult] = None
        self._current_positions: Dict[str, float] = {}  # symbol -> quantity
        self._current_prices: Dict[str, float] = {}     # symbol -> price
        self._cash_balance: float = self.config.total_capital
        self._returns_history: Dict[str, pd.Series] = {}

    def add_asset(self, asset: AssetConfig) -> None:
        """Add a new asset to the portfolio."""
        self.config.assets.append(asset)
        if asset.enabled:
            self.asset_engines[asset.symbol] = SmartTradingEngine(
                config=self.engine_config,
                data_dir=str(self.data_dir / "engines" / asset.symbol.replace("/", "_")),
            )
        logger.info(f"Added asset {asset.symbol} to portfolio")

    def remove_asset(self, symbol: str) -> None:
        """Remove an asset from the portfolio."""
        self.config.assets = [a for a in self.config.assets if a.symbol != symbol]
        if symbol in self.asset_engines:
            del self.asset_engines[symbol]
        logger.info(f"Removed asset {symbol} from portfolio")

    def update_market_data(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        current_price: float,
    ) -> None:
        """Update market data for an asset."""
        self._current_prices[symbol] = current_price

        # Calculate returns for portfolio optimization
        if len(ohlcv) >= 2:
            returns = ohlcv["close"].pct_change().dropna()
            self._returns_history[symbol] = returns

    def analyze_portfolio(
        self,
        market_data: Dict[str, pd.DataFrame],
    ) -> PortfolioDecision:
        """
        Analyze entire portfolio and generate decisions.

        Args:
            market_data: Dictionary mapping symbol to OHLCV DataFrame

        Returns:
            PortfolioDecision with allocation and trade recommendations
        """
        timestamp = datetime.now()
        asset_decisions: List[AssetDecision] = []
        trades_to_execute: List[Dict[str, Any]] = []

        # Update prices and returns from market data
        for symbol, ohlcv in market_data.items():
            if len(ohlcv) > 0:
                self._current_prices[symbol] = float(ohlcv["close"].iloc[-1])
                self._returns_history[symbol] = ohlcv["close"].pct_change().dropna()

        # 1. Calculate optimal portfolio allocation
        returns_df = self._build_returns_dataframe()
        if len(returns_df.columns) >= 2 and len(returns_df) >= self.config.min_history_days:
            # Apply correlation filter if enabled
            if self.config.use_correlation_filter:
                returns_df = self._filter_correlated_assets(returns_df)

            # Run portfolio optimization
            self._current_allocation = self.portfolio_optimizer.optimize(
                returns_df,
                method=self.config.optimization_method,
            )
            target_weights = self._current_allocation.weights
        else:
            # Equal weight if not enough data
            enabled_assets = [a.symbol for a in self.config.assets if a.enabled and a.symbol in market_data]
            target_weights = {s: 1.0 / len(enabled_assets) for s in enabled_assets}

        # 2. Analyze each asset with smart engine
        for asset in self.config.assets:
            if not asset.enabled or asset.symbol not in market_data:
                continue

            symbol = asset.symbol
            ohlcv = market_data[symbol]

            # Get smart engine decision
            if symbol in self.asset_engines and len(ohlcv) >= 50:
                try:
                    decision = self.asset_engines[symbol].analyze(ohlcv)
                except Exception as e:
                    logger.warning(f"Smart engine analysis failed for {symbol}: {e}")
                    decision = TradingDecision(
                        action="FLAT",
                        confidence=0.0,
                        source="error",
                        regime="unknown",
                        regime_confidence=0.0,
                        ml_probability=0.33,
                        strategy_signal="FLAT",
                        strategy_confidence=0.0,
                        position_size_multiplier=0.0,
                        reasoning=[f"Analysis error: {e}"],
                    )
            else:
                decision = TradingDecision(
                    action="FLAT",
                    confidence=0.0,
                    source="insufficient_data",
                    regime="unknown",
                    regime_confidence=0.0,
                    ml_probability=0.33,
                    strategy_signal="FLAT",
                    strategy_confidence=0.0,
                    position_size_multiplier=0.0,
                    reasoning=["Insufficient data for analysis"],
                )

            # Calculate target allocation
            weight = target_weights.get(symbol, 0.0)

            # Adjust weight based on smart engine decision
            adjusted_weight = self._adjust_weight_by_signal(
                weight, decision.action, decision.confidence, decision.position_size_multiplier
            )

            # Enforce min/max weight constraints
            adjusted_weight = max(asset.min_weight, min(asset.max_weight, adjusted_weight))

            target_value = self._calculate_portfolio_value() * adjusted_weight
            current_value = self._get_position_value(symbol)
            current_price = self._current_prices.get(symbol, 0)
            target_quantity = target_value / current_price if current_price > 0 else 0

            asset_decision = AssetDecision(
                symbol=symbol,
                action=decision.action,
                weight=adjusted_weight,
                target_value=target_value,
                current_value=current_value,
                target_quantity=target_quantity,
                confidence=decision.confidence,
                regime=decision.regime,
                reasoning=decision.reasoning,
            )
            asset_decisions.append(asset_decision)

        # 3. Normalize weights to sum to 1
        total_weight = sum(d.weight for d in asset_decisions)
        if total_weight > 0:
            for d in asset_decisions:
                d.weight = d.weight / total_weight
                d.target_value = self._calculate_portfolio_value() * d.weight
                if self._current_prices.get(d.symbol, 0) > 0:
                    d.target_quantity = d.target_value / self._current_prices[d.symbol]

        # 4. Determine rebalancing trades
        current_weights = self._get_current_weights()
        target_weights_final = {d.symbol: d.weight for d in asset_decisions}

        rebalance_info = self.portfolio_optimizer.suggest_rebalancing(
            current_weights=current_weights,
            target_weights=target_weights_final,
            threshold=self.config.rebalance_threshold,
            current_prices=self._current_prices,
            portfolio_value=self._calculate_portfolio_value(),
        )

        rebalance_needed = rebalance_info["needs_rebalancing"]

        if rebalance_needed:
            for trade in rebalance_info["trades"]:
                trades_to_execute.append({
                    "symbol": trade["asset"],
                    "action": trade["action"],
                    "weight_change": trade["weight_change"],
                    "amount_usd": trade.get("amount_usd", 0),
                    "quantity": trade.get("quantity", 0),
                    "price": self._current_prices.get(trade["asset"], 0),
                })

        # 5. Calculate portfolio metrics
        portfolio_metrics = {}
        if self._current_allocation:
            portfolio_metrics = {
                "expected_return": self._current_allocation.metrics.expected_return,
                "volatility": self._current_allocation.metrics.volatility,
                "sharpe_ratio": self._current_allocation.metrics.sharpe_ratio,
                "diversification_ratio": self._current_allocation.metrics.diversification_ratio,
                "effective_n": self._current_allocation.metrics.effective_n,
            }

        return PortfolioDecision(
            timestamp=timestamp,
            asset_decisions=asset_decisions,
            allocation=target_weights_final,
            portfolio_metrics=portfolio_metrics,
            rebalance_needed=rebalance_needed,
            trades_to_execute=trades_to_execute,
            total_value=self._calculate_portfolio_value(),
            cash_balance=self._cash_balance,
        )

    def execute_rebalance(self, decision: PortfolioDecision) -> Dict[str, Any]:
        """
        Execute rebalancing trades (paper trading).

        Args:
            decision: Portfolio decision with trades to execute

        Returns:
            Execution results
        """
        if not decision.rebalance_needed:
            return {"status": "no_rebalance_needed"}

        executed_trades = []
        portfolio_value = self._calculate_portfolio_value()

        for trade in decision.trades_to_execute:
            symbol = trade["symbol"]
            action = trade["action"]
            quantity = trade["quantity"]
            price = trade["price"]

            # Validate order before execution
            if HAS_ORDER_VALIDATOR:
                validation = validate_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    portfolio_value=portfolio_value,
                    cash_balance=self._cash_balance,
                    current_position=self._current_positions.get(symbol, 0),
                )
                if not validation.is_valid:
                    logger.warning(f"Order rejected for {symbol}: {validation.message}")
                    executed_trades.append({
                        "symbol": symbol,
                        "action": action,
                        "status": "rejected",
                        "reason": validation.message,
                    })
                    continue
                # Use adjusted quantity if provided
                if validation.adjusted_quantity is not None:
                    quantity = validation.adjusted_quantity
                    logger.info(f"Order quantity adjusted for {symbol}: {quantity}")

            # Update data freshness
            if HAS_DATA_FRESHNESS:
                asset_cfg = next((a for a in self.config.assets if a.symbol == symbol), None)
                market_type = asset_cfg.asset_type if asset_cfg else "crypto"
                update_data(symbol, price, market_type)

            if action == "BUY":
                # Deduct cash, add position
                cost = quantity * price
                if cost <= self._cash_balance:
                    self._cash_balance -= cost
                    self._current_positions[symbol] = self._current_positions.get(symbol, 0) + quantity
                    executed_trades.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": quantity,
                        "price": price,
                        "cost": cost,
                        "status": "executed",
                    })
                    logger.info(f"Executed BUY {quantity:.6f} {symbol} @ ${price:,.2f}")
                else:
                    executed_trades.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "status": "insufficient_cash",
                    })
            elif action == "SELL":
                # Add cash, reduce position
                current_qty = self._current_positions.get(symbol, 0)
                sell_qty = min(quantity, current_qty)
                if sell_qty > 0:
                    proceeds = sell_qty * price
                    self._cash_balance += proceeds
                    self._current_positions[symbol] = current_qty - sell_qty
                    executed_trades.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": sell_qty,
                        "price": price,
                        "proceeds": proceeds,
                        "status": "executed",
                    })
                    logger.info(f"Executed SELL {sell_qty:.6f} {symbol} @ ${price:,.2f}")

        # Record to database and persistent data store
        for trade_info in executed_trades:
            if trade_info.get("status") == "executed":
                # Find the asset decision for this trade
                asset_decision = next(
                    (ad for ad in decision.asset_decisions if ad.symbol == trade_info["symbol"]),
                    None
                )
                regime = asset_decision.regime if asset_decision else "unknown"
                confidence = asset_decision.confidence if asset_decision else 0
                signal = asset_decision.action if asset_decision else "FLAT"

                # Record to database
                try:
                    trade_record = Trade(
                        symbol=trade_info["symbol"],
                        direction="LONG" if trade_info["action"] == "BUY" else "SHORT",
                        entry_time=datetime.now(),
                        entry_price=trade_info["price"],
                        size=trade_info["quantity"],
                        pnl=0,  # Will be calculated on exit
                        pnl_pct=0,
                        strategy="portfolio_rebalance",
                        regime=regime,
                    )
                    self.database.insert_trade(trade_record)
                except Exception as e:
                    logger.warning(f"Failed to record trade to database: {e}")

                # Record to persistent data store (for Trade History UI)
                if HAS_DATA_STORE:
                    try:
                        # Derive market type from asset config
                        asset_cfg = next(
                            (a for a in self.config.assets if a.symbol == trade_info["symbol"]),
                            None
                        )
                        market_type = asset_cfg.asset_type if asset_cfg else "unknown"
                        store_trade(
                            symbol=trade_info["symbol"],
                            action=trade_info["action"],
                            quantity=trade_info["quantity"],
                            price=trade_info["price"],
                            market=market_type,
                            regime=regime,
                            confidence=confidence,
                            signal=signal,
                            strategy="portfolio_rebalance",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to record trade to data store: {e}")

        return {
            "status": "executed",
            "trades": executed_trades,
            "new_cash_balance": self._cash_balance,
            "portfolio_value": self._calculate_portfolio_value(),
        }

    def _build_returns_dataframe(self) -> pd.DataFrame:
        """Build returns DataFrame for portfolio optimization."""
        if not self._returns_history:
            return pd.DataFrame()

        # Align all returns series
        returns_dict = {}
        min_length = min(len(s) for s in self._returns_history.values())

        for symbol, returns in self._returns_history.items():
            returns_dict[symbol] = returns.tail(min_length).values

        return pd.DataFrame(returns_dict)

    def _filter_correlated_assets(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out highly correlated assets."""
        if len(returns_df.columns) <= 2:
            return returns_df

        corr = returns_df.corr()
        to_remove = set()

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > self.config.max_correlation:
                    # Remove the asset with lower returns
                    asset1, asset2 = corr.columns[i], corr.columns[j]
                    mean1 = returns_df[asset1].mean()
                    mean2 = returns_df[asset2].mean()
                    to_remove.add(asset1 if mean1 < mean2 else asset2)
                    logger.info(f"Filtering correlated asset: {to_remove}")

        return returns_df.drop(columns=list(to_remove))

    def _adjust_weight_by_signal(
        self,
        base_weight: float,
        action: str,
        confidence: float,
        position_multiplier: float,
    ) -> float:
        """Adjust target weight based on trading signal."""
        # Reduce weight for FLAT or SHORT signals (for long-only portfolio)
        if action == "FLAT":
            return base_weight * 0.5 * (1 - confidence * 0.5)
        elif action == "SHORT":
            return base_weight * 0.25  # Minimal allocation for bearish assets
        else:  # LONG
            return base_weight * position_multiplier * min(1.5, 1 + confidence * 0.5)

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            qty * self._current_prices.get(symbol, 0)
            for symbol, qty in self._current_positions.items()
        )
        return self._cash_balance + positions_value

    def _get_position_value(self, symbol: str) -> float:
        """Get current value of a position."""
        qty = self._current_positions.get(symbol, 0)
        price = self._current_prices.get(symbol, 0)
        return qty * price

    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        total_value = self._calculate_portfolio_value()
        if total_value == 0:
            return {}

        weights = {}
        for symbol in self._current_positions:
            weights[symbol] = self._get_position_value(symbol) / total_value

        return weights

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            "total_value": self._calculate_portfolio_value(),
            "cash_balance": self._cash_balance,
            "positions": {
                symbol: {
                    "quantity": qty,
                    "price": self._current_prices.get(symbol, 0),
                    "value": qty * self._current_prices.get(symbol, 0),
                }
                for symbol, qty in self._current_positions.items()
                if qty > 0
            },
            "current_weights": self._get_current_weights(),
            "target_allocation": self._current_allocation.weights if self._current_allocation else {},
            "optimization_method": self.config.optimization_method.value,
            "assets_enabled": len([a for a in self.config.assets if a.enabled]),
        }

    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Get correlation analysis for portfolio assets."""
        returns_df = self._build_returns_dataframe()
        if returns_df.empty:
            return {"error": "No returns data available"}

        return self.portfolio_optimizer.analyze_correlations(returns_df)


def create_default_crypto_portfolio() -> MultiAssetConfig:
    """Create a default crypto portfolio configuration."""
    return MultiAssetConfig(
        assets=[
            AssetConfig(symbol="BTC/USDT", max_weight=0.40),
            AssetConfig(symbol="ETH/USDT", max_weight=0.35),
            AssetConfig(symbol="SOL/USDT", max_weight=0.25),
            AssetConfig(symbol="AVAX/USDT", max_weight=0.20),
            AssetConfig(symbol="MATIC/USDT", max_weight=0.20),
        ],
        optimization_method=OptimizationMethod.RISK_PARITY,
        rebalance_threshold=0.05,
        rebalance_frequency="weekly",
        total_capital=10000.0,
    )
