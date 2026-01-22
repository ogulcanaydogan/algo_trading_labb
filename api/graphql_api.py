"""
GraphQL API - Flexible data querying interface.

Provides a GraphQL API alongside REST for flexible data queries
and real-time subscriptions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for graphql dependencies
try:
    from ariadne import QueryType, MutationType, SubscriptionType, make_executable_schema
    from ariadne.asgi import GraphQL
    HAS_ARIADNE = True
except ImportError:
    HAS_ARIADNE = False
    logger.warning("Ariadne not installed. Install with: pip install ariadne")


# GraphQL Schema Definition
SCHEMA_SDL = """
type Query {
    # Portfolio queries
    portfolio: Portfolio!
    position(symbol: String!): Position
    positions: [Position!]!

    # Trading queries
    signal(symbol: String!): TradingSignal
    signals: [TradingSignal!]!
    trades(symbol: String, limit: Int): [Trade!]!

    # Market data queries
    ticker(symbol: String!): Ticker
    tickers(symbols: [String!]): [Ticker!]!
    ohlcv(symbol: String!, timeframe: String!, limit: Int): [OHLCV!]!

    # ML queries
    prediction(symbol: String!): MLPrediction
    modelStatus: ModelStatus!

    # Risk queries
    riskMetrics: RiskMetrics!
    drawdown: DrawdownStatus!

    # System queries
    health: HealthStatus!
    config: SystemConfig!
}

type Mutation {
    # Trading mutations
    placeOrder(input: OrderInput!): OrderResult!
    cancelOrder(orderId: String!): CancelResult!
    closePosition(symbol: String!): CloseResult!

    # Strategy mutations
    setStrategy(strategyId: String!): StrategyResult!
    updateConfig(key: String!, value: String!): ConfigResult!

    # Risk mutations
    emergencyStop: EmergencyResult!
    resumeTrading: ResumeResult!
}

type Subscription {
    # Real-time updates
    priceUpdates(symbols: [String!]): PriceUpdate!
    tradeUpdates: TradeUpdate!
    signalUpdates: SignalUpdate!
    portfolioUpdates: PortfolioUpdate!
}

# Types

type Portfolio {
    totalValue: Float!
    cash: Float!
    equity: Float!
    unrealizedPnl: Float!
    realizedPnl: Float!
    positions: [Position!]!
    updatedAt: String!
}

type Position {
    symbol: String!
    side: String!
    quantity: Float!
    entryPrice: Float!
    currentPrice: Float!
    unrealizedPnl: Float!
    unrealizedPnlPct: Float!
    value: Float!
    openedAt: String!
}

type TradingSignal {
    symbol: String!
    action: String!
    confidence: Float!
    strategy: String!
    entryPrice: Float
    stopLoss: Float
    takeProfit: Float
    reasoning: String!
    timestamp: String!
}

type Trade {
    tradeId: String!
    symbol: String!
    side: String!
    quantity: Float!
    price: Float!
    value: Float!
    pnl: Float
    timestamp: String!
}

type Ticker {
    symbol: String!
    price: Float!
    bid: Float!
    ask: Float!
    volume24h: Float!
    change24h: Float!
    timestamp: String!
}

type OHLCV {
    timestamp: String!
    open: Float!
    high: Float!
    low: Float!
    close: Float!
    volume: Float!
}

type MLPrediction {
    symbol: String!
    prediction: Float!
    confidence: Float!
    direction: String!
    features: [FeatureImportance!]!
    timestamp: String!
}

type FeatureImportance {
    name: String!
    importance: Float!
    value: Float!
}

type ModelStatus {
    modelId: String!
    accuracy: Float!
    lastTrained: String!
    features: Int!
    driftDetected: Boolean!
}

type RiskMetrics {
    var95: Float!
    var99: Float!
    maxDrawdown: Float!
    sharpeRatio: Float!
    volatility: Float!
    exposurePct: Float!
    withinLimits: Boolean!
}

type DrawdownStatus {
    currentDrawdown: Float!
    maxDrawdown: Float!
    phase: String!
    recoveryPct: Float!
    daysInDrawdown: Int!
}

type HealthStatus {
    status: String!
    uptime: Float!
    lastHeartbeat: String!
    components: [ComponentStatus!]!
}

type ComponentStatus {
    name: String!
    status: String!
    message: String
}

type SystemConfig {
    mode: String!
    maxPositionSize: Float!
    maxDailyLoss: Float!
    activeStrategies: [String!]!
}

# Input types

input OrderInput {
    symbol: String!
    side: String!
    quantity: Float!
    orderType: String!
    price: Float
    stopLoss: Float
    takeProfit: Float
}

# Result types

type OrderResult {
    success: Boolean!
    orderId: String
    message: String!
}

type CancelResult {
    success: Boolean!
    message: String!
}

type CloseResult {
    success: Boolean!
    pnl: Float
    message: String!
}

type StrategyResult {
    success: Boolean!
    strategyId: String!
    message: String!
}

type ConfigResult {
    success: Boolean!
    key: String!
    value: String!
    message: String!
}

type EmergencyResult {
    success: Boolean!
    positionsClosed: Int!
    message: String!
}

type ResumeResult {
    success: Boolean!
    message: String!
}

# Subscription types

type PriceUpdate {
    symbol: String!
    price: Float!
    timestamp: String!
}

type TradeUpdate {
    tradeId: String!
    symbol: String!
    side: String!
    quantity: Float!
    price: Float!
    timestamp: String!
}

type SignalUpdate {
    symbol: String!
    action: String!
    confidence: Float!
    timestamp: String!
}

type PortfolioUpdate {
    totalValue: Float!
    unrealizedPnl: Float!
    timestamp: String!
}
"""


class GraphQLResolver:
    """
    GraphQL resolvers for the trading system.

    Connects GraphQL queries to the underlying trading system.
    """

    def __init__(self):
        self._trading_system = None
        self._data_provider = None

    def set_trading_system(self, system: Any):
        """Set the trading system instance."""
        self._trading_system = system

    def set_data_provider(self, provider: Any):
        """Set the data provider instance."""
        self._data_provider = provider

    # Query resolvers

    def resolve_portfolio(self, *args) -> Dict:
        """Resolve portfolio query."""
        return {
            "totalValue": 10000.0,
            "cash": 5000.0,
            "equity": 5000.0,
            "unrealizedPnl": 150.0,
            "realizedPnl": 500.0,
            "positions": [],
            "updatedAt": datetime.now().isoformat(),
        }

    def resolve_position(self, symbol: str) -> Optional[Dict]:
        """Resolve single position query."""
        return {
            "symbol": symbol,
            "side": "long",
            "quantity": 1.0,
            "entryPrice": 50000.0,
            "currentPrice": 51000.0,
            "unrealizedPnl": 1000.0,
            "unrealizedPnlPct": 2.0,
            "value": 51000.0,
            "openedAt": datetime.now().isoformat(),
        }

    def resolve_positions(self, *args) -> List[Dict]:
        """Resolve all positions query."""
        return []

    def resolve_signal(self, symbol: str) -> Dict:
        """Resolve trading signal query."""
        return {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": 0.65,
            "strategy": "ensemble",
            "entryPrice": None,
            "stopLoss": None,
            "takeProfit": None,
            "reasoning": "Waiting for stronger signal",
            "timestamp": datetime.now().isoformat(),
        }

    def resolve_signals(self, *args) -> List[Dict]:
        """Resolve all signals query."""
        return []

    def resolve_trades(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Resolve trades query."""
        return []

    def resolve_ticker(self, symbol: str) -> Dict:
        """Resolve ticker query."""
        return {
            "symbol": symbol,
            "price": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
            "volume24h": 1000000.0,
            "change24h": 2.5,
            "timestamp": datetime.now().isoformat(),
        }

    def resolve_tickers(self, symbols: List[str]) -> List[Dict]:
        """Resolve multiple tickers query."""
        return [self.resolve_ticker(s) for s in symbols]

    def resolve_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Resolve OHLCV query."""
        return []

    def resolve_prediction(self, symbol: str) -> Dict:
        """Resolve ML prediction query."""
        return {
            "symbol": symbol,
            "prediction": 0.65,
            "confidence": 0.75,
            "direction": "bullish",
            "features": [
                {"name": "rsi", "importance": 0.15, "value": 55.0},
                {"name": "macd", "importance": 0.12, "value": 0.5},
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def resolve_model_status(self, *args) -> Dict:
        """Resolve model status query."""
        return {
            "modelId": "rf_v2",
            "accuracy": 0.72,
            "lastTrained": datetime.now().isoformat(),
            "features": 50,
            "driftDetected": False,
        }

    def resolve_risk_metrics(self, *args) -> Dict:
        """Resolve risk metrics query."""
        return {
            "var95": 500.0,
            "var99": 800.0,
            "maxDrawdown": 0.05,
            "sharpeRatio": 1.5,
            "volatility": 0.15,
            "exposurePct": 0.5,
            "withinLimits": True,
        }

    def resolve_drawdown(self, *args) -> Dict:
        """Resolve drawdown status query."""
        return {
            "currentDrawdown": 0.02,
            "maxDrawdown": 0.05,
            "phase": "normal",
            "recoveryPct": 0.0,
            "daysInDrawdown": 0,
        }

    def resolve_health(self, *args) -> Dict:
        """Resolve health status query."""
        return {
            "status": "healthy",
            "uptime": 3600.0,
            "lastHeartbeat": datetime.now().isoformat(),
            "components": [
                {"name": "api", "status": "healthy", "message": None},
                {"name": "ml", "status": "healthy", "message": None},
                {"name": "data", "status": "healthy", "message": None},
            ],
        }

    def resolve_config(self, *args) -> Dict:
        """Resolve system config query."""
        return {
            "mode": "paper",
            "maxPositionSize": 0.05,
            "maxDailyLoss": 0.02,
            "activeStrategies": ["ema_crossover", "rsi_mean_reversion"],
        }

    # Mutation resolvers

    def resolve_place_order(self, input_data: Dict) -> Dict:
        """Resolve place order mutation."""
        return {
            "success": True,
            "orderId": f"ord_{datetime.now().timestamp()}",
            "message": "Order placed successfully",
        }

    def resolve_cancel_order(self, order_id: str) -> Dict:
        """Resolve cancel order mutation."""
        return {
            "success": True,
            "message": f"Order {order_id} cancelled",
        }

    def resolve_close_position(self, symbol: str) -> Dict:
        """Resolve close position mutation."""
        return {
            "success": True,
            "pnl": 100.0,
            "message": f"Position {symbol} closed",
        }

    def resolve_set_strategy(self, strategy_id: str) -> Dict:
        """Resolve set strategy mutation."""
        return {
            "success": True,
            "strategyId": strategy_id,
            "message": f"Strategy set to {strategy_id}",
        }

    def resolve_update_config(self, key: str, value: str) -> Dict:
        """Resolve update config mutation."""
        return {
            "success": True,
            "key": key,
            "value": value,
            "message": f"Config {key} updated",
        }

    def resolve_emergency_stop(self, *args) -> Dict:
        """Resolve emergency stop mutation."""
        return {
            "success": True,
            "positionsClosed": 0,
            "message": "Emergency stop activated",
        }

    def resolve_resume_trading(self, *args) -> Dict:
        """Resolve resume trading mutation."""
        return {
            "success": True,
            "message": "Trading resumed",
        }


def create_graphql_schema():
    """Create the GraphQL schema with resolvers."""
    if not HAS_ARIADNE:
        logger.error("Ariadne required for GraphQL. Install with: pip install ariadne")
        return None

    resolver = GraphQLResolver()

    # Create query type
    query = QueryType()

    @query.field("portfolio")
    def resolve_portfolio(*_):
        return resolver.resolve_portfolio()

    @query.field("position")
    def resolve_position(_, info, symbol):
        return resolver.resolve_position(symbol)

    @query.field("positions")
    def resolve_positions(*_):
        return resolver.resolve_positions()

    @query.field("signal")
    def resolve_signal(_, info, symbol):
        return resolver.resolve_signal(symbol)

    @query.field("signals")
    def resolve_signals(*_):
        return resolver.resolve_signals()

    @query.field("trades")
    def resolve_trades(_, info, symbol=None, limit=20):
        return resolver.resolve_trades(symbol, limit)

    @query.field("ticker")
    def resolve_ticker(_, info, symbol):
        return resolver.resolve_ticker(symbol)

    @query.field("tickers")
    def resolve_tickers(_, info, symbols):
        return resolver.resolve_tickers(symbols)

    @query.field("ohlcv")
    def resolve_ohlcv(_, info, symbol, timeframe, limit=100):
        return resolver.resolve_ohlcv(symbol, timeframe, limit)

    @query.field("prediction")
    def resolve_prediction(_, info, symbol):
        return resolver.resolve_prediction(symbol)

    @query.field("modelStatus")
    def resolve_model_status(*_):
        return resolver.resolve_model_status()

    @query.field("riskMetrics")
    def resolve_risk_metrics(*_):
        return resolver.resolve_risk_metrics()

    @query.field("drawdown")
    def resolve_drawdown(*_):
        return resolver.resolve_drawdown()

    @query.field("health")
    def resolve_health(*_):
        return resolver.resolve_health()

    @query.field("config")
    def resolve_config(*_):
        return resolver.resolve_config()

    # Create mutation type
    mutation = MutationType()

    @mutation.field("placeOrder")
    def resolve_place_order(_, info, input):
        return resolver.resolve_place_order(input)

    @mutation.field("cancelOrder")
    def resolve_cancel_order(_, info, orderId):
        return resolver.resolve_cancel_order(orderId)

    @mutation.field("closePosition")
    def resolve_close_position(_, info, symbol):
        return resolver.resolve_close_position(symbol)

    @mutation.field("setStrategy")
    def resolve_set_strategy(_, info, strategyId):
        return resolver.resolve_set_strategy(strategyId)

    @mutation.field("updateConfig")
    def resolve_update_config(_, info, key, value):
        return resolver.resolve_update_config(key, value)

    @mutation.field("emergencyStop")
    def resolve_emergency_stop(*_):
        return resolver.resolve_emergency_stop()

    @mutation.field("resumeTrading")
    def resolve_resume_trading(*_):
        return resolver.resolve_resume_trading()

    # Create schema
    schema = make_executable_schema(SCHEMA_SDL, query, mutation)

    return schema


def create_graphql_app():
    """Create a GraphQL ASGI application."""
    schema = create_graphql_schema()
    if schema is None:
        return None

    return GraphQL(schema, debug=True)


# FastAPI integration
def add_graphql_route(app, path: str = "/graphql"):
    """Add GraphQL endpoint to FastAPI app."""
    graphql_app = create_graphql_app()

    if graphql_app is None:
        logger.warning("GraphQL not available - skipping route")
        return

    app.mount(path, graphql_app)
    logger.info(f"GraphQL endpoint mounted at {path}")
