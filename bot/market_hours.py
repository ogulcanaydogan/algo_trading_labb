"""
Market Hours Utility.

Provides timezone-aware market hours checking for different asset classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Literal, Optional, Union
from enum import Enum
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Types of markets."""

    CRYPTO = "crypto"
    FOREX = "forex"
    STOCK = "stock"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass
class TradingHours:
    """Trading hours configuration for a market."""

    market_type: MarketType
    open_time: Optional[time] = None
    close_time: Optional[time] = None
    timezone: str = "America/New_York"
    open_days: List[int] = None  # 0=Monday, 6=Sunday
    is_24_7: bool = False
    pre_market_start: Optional[time] = None
    after_hours_end: Optional[time] = None

    def __post_init__(self):
        if self.open_days is None:
            self.open_days = [0, 1, 2, 3, 4]  # Mon-Fri default


# Predefined trading hours for major markets
MARKET_HOURS: Dict[str, TradingHours] = {
    # Crypto - 24/7
    "crypto": TradingHours(
        market_type=MarketType.CRYPTO,
        is_24_7=True,
    ),
    # Forex - 24/5 (Sunday 5pm - Friday 5pm ET)
    "forex": TradingHours(
        market_type=MarketType.FOREX,
        open_time=time(17, 0),  # Sunday 5pm
        close_time=time(17, 0),  # Friday 5pm
        timezone="America/New_York",
        open_days=[0, 1, 2, 3, 4],  # Continuous Mon-Fri
    ),
    # US Stocks (NYSE/NASDAQ)
    "stock_us": TradingHours(
        market_type=MarketType.STOCK,
        open_time=time(9, 30),
        close_time=time(16, 0),
        timezone="America/New_York",
        open_days=[0, 1, 2, 3, 4],
        pre_market_start=time(4, 0),
        after_hours_end=time(20, 0),
    ),
    # Commodities (CME/NYMEX)
    "commodity_us": TradingHours(
        market_type=MarketType.COMMODITY,
        open_time=time(18, 0),  # Sunday 6pm
        close_time=time(17, 0),  # Friday 5pm
        timezone="America/New_York",
        open_days=[0, 1, 2, 3, 4],
    ),
    # European Stocks (LSE)
    "stock_uk": TradingHours(
        market_type=MarketType.STOCK,
        open_time=time(8, 0),
        close_time=time(16, 30),
        timezone="Europe/London",
        open_days=[0, 1, 2, 3, 4],
    ),
    # Asian Stocks (Tokyo)
    "stock_japan": TradingHours(
        market_type=MarketType.STOCK,
        open_time=time(9, 0),
        close_time=time(15, 0),
        timezone="Asia/Tokyo",
        open_days=[0, 1, 2, 3, 4],
    ),
}


class MarketHoursChecker:
    """
    Checks if markets are open based on trading hours.

    Usage:
        checker = MarketHoursChecker()

        # Check if a specific market is open
        is_open = checker.is_market_open("stock_us")

        # Check if open with custom hours
        custom_hours = TradingHours(
            market_type=MarketType.STOCK,
            open_time=time(9, 30),
            close_time=time(16, 0),
            timezone="America/New_York",
        )
        is_open = checker.is_open(custom_hours)

        # Get next open/close times
        next_open = checker.next_open_time("stock_us")
        next_close = checker.next_close_time("stock_us")
    """

    def __init__(self, custom_hours: Optional[Dict[str, TradingHours]] = None):
        """
        Initialize market hours checker.

        Args:
            custom_hours: Optional custom market hours to add/override
        """
        self.market_hours = MARKET_HOURS.copy()
        if custom_hours:
            self.market_hours.update(custom_hours)

    def is_market_open(
        self,
        market_key: str,
        check_time: Optional[datetime] = None,
        include_extended: bool = False,
    ) -> bool:
        """
        Check if a predefined market is open.

        Args:
            market_key: Key from MARKET_HOURS (e.g., "stock_us", "crypto")
            check_time: Time to check (default: now)
            include_extended: Include pre-market/after-hours for stocks

        Returns:
            True if market is open
        """
        if market_key not in self.market_hours:
            logger.warning(f"Unknown market: {market_key}")
            return True  # Default to open for unknown markets

        hours = self.market_hours[market_key]
        return self.is_open(hours, check_time, include_extended)

    def is_open(
        self,
        hours: TradingHours,
        check_time: Optional[datetime] = None,
        include_extended: bool = False,
    ) -> bool:
        """
        Check if market is open based on trading hours.

        Args:
            hours: TradingHours configuration
            check_time: Time to check (default: now)
            include_extended: Include pre-market/after-hours

        Returns:
            True if market is open
        """
        # 24/7 markets are always open
        if hours.is_24_7:
            return True

        # Get current time in market timezone
        tz = ZoneInfo(hours.timezone)
        if check_time is None:
            check_time = datetime.now(tz)
        else:
            check_time = check_time.astimezone(tz)

        current_day = check_time.weekday()
        current_time = check_time.time()

        # Check if trading day
        if current_day not in hours.open_days:
            return False

        # Determine open/close times
        if include_extended and hours.pre_market_start:
            open_time = hours.pre_market_start
        else:
            open_time = hours.open_time

        if include_extended and hours.after_hours_end:
            close_time = hours.after_hours_end
        else:
            close_time = hours.close_time

        # Handle overnight sessions (e.g., forex, futures)
        if open_time and close_time and open_time > close_time:
            # Overnight session
            return current_time >= open_time or current_time <= close_time
        else:
            # Normal session
            if open_time and close_time:
                return open_time <= current_time <= close_time

        return True

    def is_extended_hours(
        self,
        market_key: str,
        check_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if we're in extended hours (pre-market or after-hours).

        Args:
            market_key: Key from MARKET_HOURS
            check_time: Time to check (default: now)

        Returns:
            True if in extended hours
        """
        if market_key not in self.market_hours:
            return False

        hours = self.market_hours[market_key]

        if hours.is_24_7:
            return False

        if not hours.pre_market_start and not hours.after_hours_end:
            return False

        tz = ZoneInfo(hours.timezone)
        if check_time is None:
            check_time = datetime.now(tz)
        else:
            check_time = check_time.astimezone(tz)

        current_day = check_time.weekday()
        current_time = check_time.time()

        if current_day not in hours.open_days:
            return False

        # Pre-market check
        if hours.pre_market_start and hours.open_time:
            if hours.pre_market_start <= current_time < hours.open_time:
                return True

        # After-hours check
        if hours.after_hours_end and hours.close_time:
            if hours.close_time < current_time <= hours.after_hours_end:
                return True

        return False

    def next_open_time(
        self,
        market_key: str,
        from_time: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """
        Get the next market open time.

        Args:
            market_key: Key from MARKET_HOURS
            from_time: Start time to search from (default: now)

        Returns:
            Next open datetime or None if 24/7
        """
        if market_key not in self.market_hours:
            return None

        hours = self.market_hours[market_key]

        if hours.is_24_7:
            return None

        tz = ZoneInfo(hours.timezone)
        if from_time is None:
            from_time = datetime.now(tz)
        else:
            from_time = from_time.astimezone(tz)

        # Search next 7 days
        for days_ahead in range(8):
            check_date = from_time.date() + timedelta(days=days_ahead)
            check_day = check_date.weekday()

            if check_day in hours.open_days:
                if hours.open_time:
                    open_dt = datetime.combine(check_date, hours.open_time, tzinfo=tz)

                    # If today and already past open, skip
                    if days_ahead == 0 and open_dt <= from_time:
                        continue

                    return open_dt

        return None

    def next_close_time(
        self,
        market_key: str,
        from_time: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """
        Get the next market close time.

        Args:
            market_key: Key from MARKET_HOURS
            from_time: Start time to search from (default: now)

        Returns:
            Next close datetime or None if 24/7
        """
        if market_key not in self.market_hours:
            return None

        hours = self.market_hours[market_key]

        if hours.is_24_7:
            return None

        tz = ZoneInfo(hours.timezone)
        if from_time is None:
            from_time = datetime.now(tz)
        else:
            from_time = from_time.astimezone(tz)

        # Search next 7 days
        for days_ahead in range(8):
            check_date = from_time.date() + timedelta(days=days_ahead)
            check_day = check_date.weekday()

            if check_day in hours.open_days:
                if hours.close_time:
                    close_dt = datetime.combine(check_date, hours.close_time, tzinfo=tz)

                    # If today and already past close, skip
                    if days_ahead == 0 and close_dt <= from_time:
                        continue

                    return close_dt

        return None

    def time_until_open(self, market_key: str) -> Optional[timedelta]:
        """Get time remaining until market opens."""
        next_open = self.next_open_time(market_key)
        if next_open:
            tz = ZoneInfo(self.market_hours[market_key].timezone)
            now = datetime.now(tz)
            return next_open - now
        return None

    def time_until_close(self, market_key: str) -> Optional[timedelta]:
        """Get time remaining until market closes."""
        if not self.is_market_open(market_key):
            return None

        next_close = self.next_close_time(market_key)
        if next_close:
            tz = ZoneInfo(self.market_hours[market_key].timezone)
            now = datetime.now(tz)
            return next_close - now
        return None

    def get_market_status(self, market_key: str) -> Dict:
        """
        Get comprehensive market status.

        Returns:
            Dict with status info including open/close times
        """
        is_open = self.is_market_open(market_key)
        is_extended = self.is_extended_hours(market_key)

        status = {
            "market": market_key,
            "is_open": is_open,
            "is_extended_hours": is_extended,
            "next_open": None,
            "next_close": None,
            "time_until_open": None,
            "time_until_close": None,
        }

        if market_key in self.market_hours:
            hours = self.market_hours[market_key]
            status["timezone"] = hours.timezone
            status["is_24_7"] = hours.is_24_7

            if not hours.is_24_7:
                next_open = self.next_open_time(market_key)
                next_close = self.next_close_time(market_key)

                if next_open:
                    status["next_open"] = next_open.isoformat()
                    time_until = self.time_until_open(market_key)
                    if time_until:
                        status["time_until_open"] = str(time_until)

                if next_close and is_open:
                    status["next_close"] = next_close.isoformat()
                    time_until = self.time_until_close(market_key)
                    if time_until:
                        status["time_until_close"] = str(time_until)

        return status


def get_market_key_for_symbol(symbol: str) -> str:
    """
    Determine the market key for a given symbol.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT", "AAPL", "EUR/USD")

    Returns:
        Market key string (e.g., "crypto", "stock_us", "forex")
    """
    symbol_upper = symbol.upper()

    # Crypto pairs (contain USDT, BTC, ETH, etc.)
    crypto_bases = ["USDT", "BTC", "ETH", "BUSD", "USDC"]
    if any(base in symbol_upper for base in crypto_bases):
        return "crypto"

    # Forex pairs (currency pairs like EUR/USD)
    forex_currencies = ["EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
    if "/" in symbol and any(curr in symbol_upper for curr in forex_currencies):
        if "USD" in symbol_upper or any(curr in symbol_upper for curr in forex_currencies):
            return "forex"

    # Commodities
    commodity_symbols = ["XAU", "XAG", "USOIL", "UKOIL", "NATGAS", "COPPER", "PLATINUM"]
    if any(comm in symbol_upper for comm in commodity_symbols):
        return "commodity_us"

    # UK Stocks
    if symbol_upper.endswith(".L"):
        return "stock_uk"

    # Japanese stocks
    if symbol_upper.endswith(".T"):
        return "stock_japan"

    # Default to US stocks
    return "stock_us"


# Convenience functions
def is_market_open(symbol: str, include_extended: bool = False) -> bool:
    """Check if market is open for a given symbol."""
    checker = MarketHoursChecker()
    market_key = get_market_key_for_symbol(symbol)
    return checker.is_market_open(market_key, include_extended=include_extended)


def get_market_status(symbol: str) -> Dict:
    """Get market status for a given symbol."""
    checker = MarketHoursChecker()
    market_key = get_market_key_for_symbol(symbol)
    return checker.get_market_status(market_key)
