"""
Multi-market data layer for unified access to crypto, stocks, and commodities.
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
]
