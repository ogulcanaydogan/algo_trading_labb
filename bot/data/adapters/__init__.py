"""
Data provider adapters for multi-market trading platform.
"""

from .base import DataAdapter
from .yahoo_adapter import YahooAdapter

# CCXT adapter (optional, requires ccxt package)
try:
    from .ccxt_adapter import CCXTAdapter

    HAS_CCXT = True
except ImportError:
    CCXTAdapter = None
    HAS_CCXT = False

__all__ = ["DataAdapter", "YahooAdapter", "CCXTAdapter", "HAS_CCXT"]
