"""
Data freshness monitoring module.

Tracks data age and alerts when data becomes stale.
Prevents trading on outdated market data.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class FreshnessStatus(Enum):
    """Status of data freshness."""

    FRESH = "fresh"  # Data is current
    STALE = "stale"  # Data is old but usable
    EXPIRED = "expired"  # Data too old to use
    UNKNOWN = "unknown"  # No data received yet


@dataclass
class DataFreshnessConfig:
    """Configuration for data freshness thresholds."""

    # Crypto markets (24/7)
    crypto_fresh_seconds: int = 60  # Fresh if < 1 minute old
    crypto_stale_seconds: int = 300  # Stale if < 5 minutes old
    crypto_expired_seconds: int = 900  # Expired if > 15 minutes old

    # Stock markets (market hours only)
    stock_fresh_seconds: int = 120  # Fresh if < 2 minutes old
    stock_stale_seconds: int = 600  # Stale if < 10 minutes old
    stock_expired_seconds: int = 1800  # Expired if > 30 minutes old

    # Commodity markets
    commodity_fresh_seconds: int = 120
    commodity_stale_seconds: int = 600
    commodity_expired_seconds: int = 1200


@dataclass
class DataFreshnessResult:
    """Result of freshness check for a symbol."""

    symbol: str
    status: FreshnessStatus
    age_seconds: float
    last_update: Optional[datetime]
    message: str
    is_tradeable: bool


@dataclass
class SymbolData:
    """Tracked data for a symbol."""

    last_update: datetime
    last_price: float
    market_type: str
    update_count: int = 0


class DataFreshnessMonitor:
    """
    Monitors data freshness across all symbols.

    Features:
    - Per-symbol freshness tracking
    - Market-type aware thresholds
    - Trading gate (block trades on stale data)
    - Alerts on data staleness
    """

    def __init__(self, config: Optional[DataFreshnessConfig] = None):
        self.config = config or DataFreshnessConfig()
        self._symbol_data: Dict[str, SymbolData] = {}
        self._alerts_sent: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=5)

    def update(self, symbol: str, price: float, market_type: str = "crypto"):
        """
        Update data timestamp for a symbol.

        Args:
            symbol: Trading symbol
            price: Latest price
            market_type: Market type (crypto, stock, commodity)
        """
        now = datetime.now()

        if symbol in self._symbol_data:
            self._symbol_data[symbol].last_update = now
            self._symbol_data[symbol].last_price = price
            self._symbol_data[symbol].update_count += 1
        else:
            self._symbol_data[symbol] = SymbolData(
                last_update=now,
                last_price=price,
                market_type=market_type,
            )

    def check_freshness(self, symbol: str) -> DataFreshnessResult:
        """
        Check data freshness for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            DataFreshnessResult with status and details
        """
        if symbol not in self._symbol_data:
            return DataFreshnessResult(
                symbol=symbol,
                status=FreshnessStatus.UNKNOWN,
                age_seconds=float("inf"),
                last_update=None,
                message=f"No data received for {symbol}",
                is_tradeable=False,
            )

        data = self._symbol_data[symbol]
        now = datetime.now()
        age = (now - data.last_update).total_seconds()

        # Get thresholds based on market type
        if data.market_type == "crypto":
            fresh_threshold = self.config.crypto_fresh_seconds
            stale_threshold = self.config.crypto_stale_seconds
            expired_threshold = self.config.crypto_expired_seconds
        elif data.market_type == "stock":
            fresh_threshold = self.config.stock_fresh_seconds
            stale_threshold = self.config.stock_stale_seconds
            expired_threshold = self.config.stock_expired_seconds
        else:  # commodity
            fresh_threshold = self.config.commodity_fresh_seconds
            stale_threshold = self.config.commodity_stale_seconds
            expired_threshold = self.config.commodity_expired_seconds

        # Determine status
        if age < fresh_threshold:
            status = FreshnessStatus.FRESH
            message = f"Data is fresh ({age:.0f}s old)"
            is_tradeable = True
        elif age < stale_threshold:
            status = FreshnessStatus.STALE
            message = f"Data is stale ({age:.0f}s old) - trading with caution"
            is_tradeable = True
        elif age < expired_threshold:
            status = FreshnessStatus.STALE
            message = f"Data is very stale ({age:.0f}s old) - reduced position sizes recommended"
            is_tradeable = True
        else:
            status = FreshnessStatus.EXPIRED
            message = f"Data expired ({age:.0f}s old) - trading blocked"
            is_tradeable = False
            self._send_alert(symbol, age)

        return DataFreshnessResult(
            symbol=symbol,
            status=status,
            age_seconds=age,
            last_update=data.last_update,
            message=message,
            is_tradeable=is_tradeable,
        )

    def check_all(self) -> Dict[str, DataFreshnessResult]:
        """Check freshness for all tracked symbols."""
        return {symbol: self.check_freshness(symbol) for symbol in self._symbol_data}

    def is_tradeable(self, symbol: str) -> bool:
        """Quick check if symbol data is tradeable."""
        result = self.check_freshness(symbol)
        return result.is_tradeable

    def get_tradeable_symbols(self) -> List[str]:
        """Get list of symbols with fresh enough data to trade."""
        return [symbol for symbol in self._symbol_data if self.check_freshness(symbol).is_tradeable]

    def get_stale_symbols(self) -> List[str]:
        """Get list of symbols with stale or expired data."""
        return [
            symbol
            for symbol in self._symbol_data
            if not self.check_freshness(symbol).status == FreshnessStatus.FRESH
        ]

    def _send_alert(self, symbol: str, age: float):
        """Send alert for stale data (with cooldown)."""
        now = datetime.now()

        # Check cooldown
        if symbol in self._alerts_sent:
            if now - self._alerts_sent[symbol] < self._alert_cooldown:
                return

        logger.warning(f"STALE DATA ALERT: {symbol} data is {age:.0f}s old")
        self._alerts_sent[symbol] = now

    def get_summary(self) -> Dict:
        """Get summary of data freshness across all symbols."""
        results = self.check_all()

        fresh_count = sum(1 for r in results.values() if r.status == FreshnessStatus.FRESH)
        stale_count = sum(1 for r in results.values() if r.status == FreshnessStatus.STALE)
        expired_count = sum(1 for r in results.values() if r.status == FreshnessStatus.EXPIRED)
        unknown_count = sum(1 for r in results.values() if r.status == FreshnessStatus.UNKNOWN)

        avg_age = sum(
            r.age_seconds for r in results.values() if r.age_seconds < float("inf")
        ) / max(len(results), 1)

        return {
            "total_symbols": len(self._symbol_data),
            "fresh": fresh_count,
            "stale": stale_count,
            "expired": expired_count,
            "unknown": unknown_count,
            "average_age_seconds": avg_age,
            "tradeable_count": fresh_count + stale_count,
            "symbols": {
                symbol: {
                    "status": r.status.value,
                    "age_seconds": r.age_seconds,
                    "is_tradeable": r.is_tradeable,
                }
                for symbol, r in results.items()
            },
        }

    def reset(self, symbol: Optional[str] = None):
        """Reset tracking for symbol(s)."""
        if symbol:
            self._symbol_data.pop(symbol, None)
            self._alerts_sent.pop(symbol, None)
        else:
            self._symbol_data.clear()
            self._alerts_sent.clear()


# Global monitor instance
_monitor: Optional[DataFreshnessMonitor] = None


def get_monitor() -> DataFreshnessMonitor:
    """Get or create global freshness monitor."""
    global _monitor
    if _monitor is None:
        _monitor = DataFreshnessMonitor()
    return _monitor


def update_data(symbol: str, price: float, market_type: str = "crypto"):
    """Convenience function to update data freshness."""
    get_monitor().update(symbol, price, market_type)


def is_data_fresh(symbol: str) -> bool:
    """Convenience function to check if data is tradeable."""
    return get_monitor().is_tradeable(symbol)
