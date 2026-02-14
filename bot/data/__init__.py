"""
Multi-market data layer for unified access to crypto, stocks, and commodities.

Includes:
- REST API adapters (Yahoo, CCXT)
- Real-time WebSocket streaming (Binance, Alpaca)
- Bar aggregation and event-driven architecture
"""

from .models import (
    MarketType,
    DataSource,
    NormalizedOHLCV,
    NormalizedQuote,
    SymbolInfo,
    CRYPTO_SYMBOLS,
    COMMODITY_SYMBOLS,
    STOCK_SYMBOLS,
    ALL_SYMBOLS,
    get_symbol_info,
    get_symbols_by_market,
)
from .cache import CacheManager
from .service import MarketDataService, get_market_data_service

# Real-time streaming
from .stream_manager import (
    # Core classes
    StreamManager,
    StreamManagerConfig,
    StreamingTradingEngine,
    # Stream handlers
    BinanceStreamHandler,
    AlpacaStreamHandler,
    BaseStreamHandler,
    # Bar aggregation
    BarAggregator,
    Bar,
    Tick,
    # Event types
    StreamEvent,
    EventType,
    StreamStatus,
    StreamHealth,
    AssetClass,
    # Factory functions
    create_stream_manager,
    create_crypto_stream,
    create_stock_stream,
    create_default_trading_stream,
)

# Legacy websocket stream (for compatibility)
from .websocket_stream import (
    BinanceWebSocket,
    StreamConfig,
    StreamType,
    TickerUpdate,
    TradeUpdate,
    OrderbookUpdate,
    create_binance_stream,
    create_stream_manager as create_legacy_stream_manager,
)

__all__ = [
    # Models
    "MarketType",
    "DataSource",
    "NormalizedOHLCV",
    "NormalizedQuote",
    "SymbolInfo",
    # Symbol registries
    "CRYPTO_SYMBOLS",
    "COMMODITY_SYMBOLS",
    "STOCK_SYMBOLS",
    "ALL_SYMBOLS",
    "get_symbol_info",
    "get_symbols_by_market",
    # Cache
    "CacheManager",
    # Service
    "MarketDataService",
    "get_market_data_service",
    # Real-time streaming (new)
    "StreamManager",
    "StreamManagerConfig",
    "StreamingTradingEngine",
    "BinanceStreamHandler",
    "AlpacaStreamHandler",
    "BaseStreamHandler",
    "BarAggregator",
    "Bar",
    "Tick",
    "StreamEvent",
    "EventType",
    "StreamStatus",
    "StreamHealth",
    "AssetClass",
    "create_stream_manager",
    "create_crypto_stream",
    "create_stock_stream",
    "create_default_trading_stream",
    # Legacy streaming (compatibility)
    "BinanceWebSocket",
    "StreamConfig",
    "StreamType",
    "TickerUpdate",
    "TradeUpdate",
    "OrderbookUpdate",
    "create_binance_stream",
    "create_legacy_stream_manager",
]
