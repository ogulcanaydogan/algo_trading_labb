"""
Data provider adapters for multi-market trading platform.
"""

from .base import DataAdapter
from .yahoo_adapter import YahooAdapter

__all__ = ["DataAdapter", "YahooAdapter"]
