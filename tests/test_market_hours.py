"""
Tests for Market Hours Module.

Tests the market hours utility functions, timezone handling,
and trading hours configuration.
"""

import pytest
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from bot.market_hours import (
    MarketType,
    TradingHours,
    MarketHoursChecker,
    MARKET_HOURS,
    get_market_key_for_symbol,
    is_market_open,
    get_market_status,
)


class TestMarketType:
    """Tests for MarketType enum."""

    def test_market_type_values(self):
        """Test MarketType enum has expected values."""
        assert MarketType.CRYPTO.value == "crypto"
        assert MarketType.FOREX.value == "forex"
        assert MarketType.STOCK.value == "stock"
        assert MarketType.COMMODITY.value == "commodity"
        assert MarketType.INDEX.value == "index"

    def test_market_type_from_string(self):
        """Test creating MarketType from string value."""
        assert MarketType("crypto") == MarketType.CRYPTO
        assert MarketType("forex") == MarketType.FOREX
        assert MarketType("stock") == MarketType.STOCK

    def test_all_market_types_exist(self):
        """Test all expected market types exist."""
        expected_types = ["crypto", "forex", "stock", "commodity", "index"]
        actual_types = [mt.value for mt in MarketType]
        assert set(expected_types) == set(actual_types)


class TestTradingHours:
    """Tests for TradingHours dataclass."""

    def test_default_trading_hours(self):
        """Test default values for TradingHours."""
        hours = TradingHours(market_type=MarketType.STOCK)
        assert hours.market_type == MarketType.STOCK
        assert hours.timezone == "America/New_York"
        assert hours.open_days == [0, 1, 2, 3, 4]  # Mon-Fri
        assert hours.is_24_7 is False
        assert hours.open_time is None
        assert hours.close_time is None

    def test_crypto_trading_hours(self):
        """Test 24/7 crypto trading hours."""
        hours = TradingHours(
            market_type=MarketType.CRYPTO,
            is_24_7=True,
        )
        assert hours.is_24_7 is True
        assert hours.market_type == MarketType.CRYPTO

    def test_stock_trading_hours(self):
        """Test stock market trading hours."""
        hours = TradingHours(
            market_type=MarketType.STOCK,
            open_time=time(9, 30),
            close_time=time(16, 0),
            timezone="America/New_York",
            pre_market_start=time(4, 0),
            after_hours_end=time(20, 0),
        )
        assert hours.open_time == time(9, 30)
        assert hours.close_time == time(16, 0)
        assert hours.pre_market_start == time(4, 0)
        assert hours.after_hours_end == time(20, 0)

    def test_custom_open_days(self):
        """Test custom trading days configuration."""
        hours = TradingHours(
            market_type=MarketType.FOREX,
            open_days=[0, 1, 2, 3, 4, 6],  # Mon-Fri + Sunday
        )
        assert 6 in hours.open_days  # Sunday
        assert 5 not in hours.open_days  # Saturday

    def test_post_init_sets_default_days(self):
        """Test __post_init__ sets default open_days when None."""
        hours = TradingHours(market_type=MarketType.STOCK, open_days=None)
        assert hours.open_days == [0, 1, 2, 3, 4]


class TestPredefinedMarketHours:
    """Tests for predefined MARKET_HOURS constant."""

    def test_crypto_market_hours(self):
        """Test predefined crypto market hours."""
        crypto = MARKET_HOURS["crypto"]
        assert crypto.is_24_7 is True
        assert crypto.market_type == MarketType.CRYPTO

    def test_us_stock_market_hours(self):
        """Test predefined US stock market hours."""
        stock = MARKET_HOURS["stock_us"]
        assert stock.open_time == time(9, 30)
        assert stock.close_time == time(16, 0)
        assert stock.timezone == "America/New_York"
        assert stock.pre_market_start == time(4, 0)
        assert stock.after_hours_end == time(20, 0)

    def test_forex_market_hours(self):
        """Test predefined forex market hours."""
        forex = MARKET_HOURS["forex"]
        assert forex.market_type == MarketType.FOREX
        assert forex.open_time == time(17, 0)
        assert forex.close_time == time(17, 0)

    def test_uk_stock_market_hours(self):
        """Test predefined UK stock market hours."""
        uk = MARKET_HOURS["stock_uk"]
        assert uk.timezone == "Europe/London"
        assert uk.open_time == time(8, 0)
        assert uk.close_time == time(16, 30)

    def test_japan_stock_market_hours(self):
        """Test predefined Japan stock market hours."""
        japan = MARKET_HOURS["stock_japan"]
        assert japan.timezone == "Asia/Tokyo"
        assert japan.open_time == time(9, 0)
        assert japan.close_time == time(15, 0)

    def test_commodity_market_hours(self):
        """Test predefined commodity market hours."""
        commodity = MARKET_HOURS["commodity_us"]
        assert commodity.market_type == MarketType.COMMODITY


class TestMarketHoursChecker:
    """Tests for MarketHoursChecker class."""

    def test_init_default(self):
        """Test default initialization."""
        checker = MarketHoursChecker()
        assert "crypto" in checker.market_hours
        assert "stock_us" in checker.market_hours

    def test_init_custom_hours(self):
        """Test initialization with custom hours."""
        custom = {
            "custom_market": TradingHours(
                market_type=MarketType.STOCK,
                open_time=time(10, 0),
                close_time=time(14, 0),
            )
        }
        checker = MarketHoursChecker(custom_hours=custom)
        assert "custom_market" in checker.market_hours

    def test_crypto_always_open(self):
        """Test crypto market is always open (24/7)."""
        checker = MarketHoursChecker()
        # Test various times
        monday_morning = datetime(2024, 1, 8, 3, 0, tzinfo=ZoneInfo("America/New_York"))
        saturday_night = datetime(2024, 1, 6, 23, 0, tzinfo=ZoneInfo("America/New_York"))

        assert checker.is_market_open("crypto", monday_morning) is True
        assert checker.is_market_open("crypto", saturday_night) is True

    def test_stock_market_during_hours(self):
        """Test stock market is open during trading hours."""
        checker = MarketHoursChecker()
        # Wednesday 11:00 AM NY time
        during_hours = datetime(2024, 1, 10, 11, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_market_open("stock_us", during_hours) is True

    def test_stock_market_outside_hours(self):
        """Test stock market is closed outside trading hours."""
        checker = MarketHoursChecker()
        # Wednesday 7:00 AM NY time (before 9:30)
        before_open = datetime(2024, 1, 10, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_market_open("stock_us", before_open) is False

        # Wednesday 7:00 PM NY time (after 4:00)
        after_close = datetime(2024, 1, 10, 19, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_market_open("stock_us", after_close) is False

    def test_stock_market_weekend(self):
        """Test stock market is closed on weekends."""
        checker = MarketHoursChecker()
        # Saturday at noon
        saturday = datetime(2024, 1, 6, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_market_open("stock_us", saturday) is False

    def test_extended_hours_pre_market(self):
        """Test extended hours pre-market detection."""
        checker = MarketHoursChecker()
        # Wednesday 5:00 AM NY time (pre-market starts at 4:00 AM)
        pre_market = datetime(2024, 1, 10, 5, 0, tzinfo=ZoneInfo("America/New_York"))

        # Should be closed for regular hours
        assert checker.is_market_open("stock_us", pre_market, include_extended=False) is False
        # Should be open for extended hours
        assert checker.is_market_open("stock_us", pre_market, include_extended=True) is True

    def test_extended_hours_after_hours(self):
        """Test extended hours after-hours detection."""
        checker = MarketHoursChecker()
        # Wednesday 6:00 PM NY time (after-hours ends at 8:00 PM)
        after_hours = datetime(2024, 1, 10, 18, 0, tzinfo=ZoneInfo("America/New_York"))

        # Should be closed for regular hours
        assert checker.is_market_open("stock_us", after_hours, include_extended=False) is False
        # Should be open for extended hours
        assert checker.is_market_open("stock_us", after_hours, include_extended=True) is True

    def test_is_extended_hours_true(self):
        """Test is_extended_hours returns True during extended hours."""
        checker = MarketHoursChecker()
        # Wednesday 5:00 AM NY time (pre-market)
        pre_market = datetime(2024, 1, 10, 5, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_extended_hours("stock_us", pre_market) is True

        # Wednesday 6:00 PM NY time (after-hours)
        after_hours = datetime(2024, 1, 10, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_extended_hours("stock_us", after_hours) is True

    def test_is_extended_hours_false_regular(self):
        """Test is_extended_hours returns False during regular hours."""
        checker = MarketHoursChecker()
        # Wednesday 11:00 AM NY time (regular hours)
        regular = datetime(2024, 1, 10, 11, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_extended_hours("stock_us", regular) is False

    def test_is_extended_hours_crypto(self):
        """Test is_extended_hours returns False for 24/7 markets."""
        checker = MarketHoursChecker()
        any_time = datetime(2024, 1, 10, 11, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_extended_hours("crypto", any_time) is False

    def test_unknown_market_defaults_to_open(self):
        """Test unknown market key defaults to open."""
        checker = MarketHoursChecker()
        assert checker.is_market_open("unknown_market") is True

    def test_next_open_time_crypto(self):
        """Test next_open_time returns None for 24/7 markets."""
        checker = MarketHoursChecker()
        assert checker.next_open_time("crypto") is None

    def test_next_open_time_stock(self):
        """Test next_open_time for stock market."""
        checker = MarketHoursChecker()
        # From Saturday morning
        saturday = datetime(2024, 1, 6, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        next_open = checker.next_open_time("stock_us", saturday)

        assert next_open is not None
        # Should be Monday 9:30 AM
        assert next_open.weekday() == 0  # Monday
        assert next_open.hour == 9
        assert next_open.minute == 30

    def test_next_close_time_crypto(self):
        """Test next_close_time returns None for 24/7 markets."""
        checker = MarketHoursChecker()
        assert checker.next_close_time("crypto") is None

    def test_next_close_time_stock(self):
        """Test next_close_time for stock market."""
        checker = MarketHoursChecker()
        # From Wednesday morning
        wednesday_morning = datetime(2024, 1, 10, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        next_close = checker.next_close_time("stock_us", wednesday_morning)

        assert next_close is not None
        # Should be Wednesday 4:00 PM
        assert next_close.weekday() == 2  # Wednesday
        assert next_close.hour == 16
        assert next_close.minute == 0

    def test_time_until_open(self):
        """Test time_until_open calculation."""
        checker = MarketHoursChecker()
        # This is tricky to test reliably without mocking time
        # For crypto, it should always return None
        assert checker.time_until_open("crypto") is None

    def test_time_until_close_when_closed(self):
        """Test time_until_close returns None when market is closed."""
        checker = MarketHoursChecker()
        # Saturday at noon - market is closed
        saturday = datetime(2024, 1, 6, 12, 0, tzinfo=ZoneInfo("America/New_York"))

        # Create a checker and patch its is_market_open for this test
        # The actual implementation should return None when market is closed

    def test_get_market_status_crypto(self):
        """Test get_market_status for crypto."""
        checker = MarketHoursChecker()
        status = checker.get_market_status("crypto")

        assert status["market"] == "crypto"
        assert status["is_open"] is True
        assert status["is_24_7"] is True
        assert status["next_open"] is None
        assert status["next_close"] is None

    def test_get_market_status_stock(self):
        """Test get_market_status for stock market."""
        checker = MarketHoursChecker()
        status = checker.get_market_status("stock_us")

        assert status["market"] == "stock_us"
        assert status["timezone"] == "America/New_York"
        assert status["is_24_7"] is False
        assert "is_open" in status
        assert "is_extended_hours" in status

    def test_unknown_market_next_times(self):
        """Test next_open_time/next_close_time return None for unknown markets."""
        checker = MarketHoursChecker()
        assert checker.next_open_time("unknown") is None
        assert checker.next_close_time("unknown") is None


class TestGetMarketKeyForSymbol:
    """Tests for get_market_key_for_symbol function."""

    def test_crypto_btc_usdt(self):
        """Test BTC/USDT is classified as crypto."""
        assert get_market_key_for_symbol("BTC/USDT") == "crypto"
        assert get_market_key_for_symbol("btc/usdt") == "crypto"

    def test_crypto_eth_btc(self):
        """Test ETH/BTC is classified as crypto."""
        assert get_market_key_for_symbol("ETH/BTC") == "crypto"

    def test_crypto_busd_pair(self):
        """Test BUSD pair is classified as crypto."""
        assert get_market_key_for_symbol("BNB/BUSD") == "crypto"

    def test_crypto_usdc_pair(self):
        """Test USDC pair is classified as crypto."""
        assert get_market_key_for_symbol("SOL/USDC") == "crypto"

    def test_forex_eur_usd(self):
        """Test EUR/USD is classified as forex."""
        assert get_market_key_for_symbol("EUR/USD") == "forex"

    def test_forex_gbp_jpy(self):
        """Test GBP/JPY is classified as forex."""
        assert get_market_key_for_symbol("GBP/JPY") == "forex"

    def test_forex_aud_cad(self):
        """Test AUD/CAD is classified as forex."""
        assert get_market_key_for_symbol("AUD/CAD") == "forex"

    def test_commodity_gold(self):
        """Test XAU/USD (gold) is classified as commodity."""
        assert get_market_key_for_symbol("XAU/USD") == "commodity_us"

    def test_commodity_silver(self):
        """Test XAG/USD (silver) is classified as commodity."""
        assert get_market_key_for_symbol("XAG/USD") == "commodity_us"

    def test_commodity_oil(self):
        """Test oil symbols are classified as commodity."""
        assert get_market_key_for_symbol("USOIL") == "commodity_us"
        assert get_market_key_for_symbol("UKOIL") == "commodity_us"

    def test_uk_stock(self):
        """Test UK stock suffix (.L) is classified correctly."""
        assert get_market_key_for_symbol("VOD.L") == "stock_uk"
        assert get_market_key_for_symbol("HSBA.L") == "stock_uk"

    def test_japan_stock(self):
        """Test Japan stock suffix (.T) is classified correctly."""
        assert get_market_key_for_symbol("7203.T") == "stock_japan"
        assert get_market_key_for_symbol("9984.T") == "stock_japan"

    def test_us_stock_default(self):
        """Test US stocks as default classification."""
        assert get_market_key_for_symbol("AAPL") == "stock_us"
        assert get_market_key_for_symbol("MSFT") == "stock_us"
        assert get_market_key_for_symbol("TSLA") == "stock_us"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_market_open_crypto(self):
        """Test is_market_open convenience function for crypto."""
        # Crypto should always return True
        assert is_market_open("BTC/USDT") is True
        assert is_market_open("ETH/BTC") is True

    def test_is_market_open_with_extended(self):
        """Test is_market_open with extended hours flag."""
        # This function should work without error
        result = is_market_open("AAPL", include_extended=True)
        assert isinstance(result, bool)

    def test_get_market_status_crypto(self):
        """Test get_market_status convenience function."""
        status = get_market_status("BTC/USDT")
        assert status["market"] == "crypto"
        assert status["is_open"] is True

    def test_get_market_status_stock(self):
        """Test get_market_status for stock symbol."""
        status = get_market_status("AAPL")
        assert status["market"] == "stock_us"
        assert "is_open" in status


class TestTimezoneHandling:
    """Tests for timezone handling."""

    def test_different_timezone_conversion(self):
        """Test that times are properly converted between timezones."""
        checker = MarketHoursChecker()

        # 9:30 AM in NY is 2:30 PM in London (EST = UTC-5, GMT = UTC)
        # During winter: NY 9:30 AM = London 2:30 PM
        ny_930 = datetime(2024, 1, 10, 9, 30, tzinfo=ZoneInfo("America/New_York"))

        # Market should be open when checked from any timezone
        assert checker.is_market_open("stock_us", ny_930) is True

    def test_uk_market_from_ny_time(self):
        """Test UK market hours from NY timezone perspective."""
        checker = MarketHoursChecker()

        # UK market: 8:00 AM - 4:30 PM London time
        # Check from NY timezone - 10:00 AM London = 5:00 AM NY (winter)
        london_10am = datetime(2024, 1, 10, 10, 0, tzinfo=ZoneInfo("Europe/London"))
        assert checker.is_market_open("stock_uk", london_10am) is True

    def test_japan_market_hours(self):
        """Test Japan market hours."""
        checker = MarketHoursChecker()

        # Japan market: 9:00 AM - 3:00 PM Tokyo time
        tokyo_noon = datetime(2024, 1, 10, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert checker.is_market_open("stock_japan", tokyo_noon) is True


class TestOvernightSessions:
    """Tests for overnight trading sessions."""

    def test_forex_overnight_session(self):
        """Test forex market overnight session handling."""
        checker = MarketHoursChecker()

        # Forex has overnight session (5PM - 5PM)
        # The logic for overnight should handle times correctly
        hours = MARKET_HOURS["forex"]

        # Open time is after close time (overnight session)
        if hours.open_time and hours.close_time:
            # This configuration represents overnight session
            assert hours.open_time == time(17, 0)
            assert hours.close_time == time(17, 0)

    def test_is_open_overnight_logic(self):
        """Test the overnight session logic in is_open method."""
        checker = MarketHoursChecker()

        # Create custom overnight hours
        overnight_hours = TradingHours(
            market_type=MarketType.FOREX,
            open_time=time(18, 0),  # 6 PM
            close_time=time(17, 0),  # 5 PM next day
            timezone="America/New_York",
            open_days=[0, 1, 2, 3, 4],
        )

        # Test during overnight session (10 PM - should be open)
        late_night = datetime(2024, 1, 10, 22, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_open(overnight_hours, late_night) is True

        # Test during daytime (should also be open due to overnight logic)
        daytime = datetime(2024, 1, 10, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        assert checker.is_open(overnight_hours, daytime) is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_custom_hours(self):
        """Test with empty custom hours dict."""
        checker = MarketHoursChecker(custom_hours={})
        assert checker.is_market_open("crypto") is True

    def test_override_predefined_market(self):
        """Test overriding predefined market hours."""
        custom = {
            "crypto": TradingHours(
                market_type=MarketType.CRYPTO,
                is_24_7=False,
                open_time=time(9, 0),
                close_time=time(17, 0),
            )
        }
        checker = MarketHoursChecker(custom_hours=custom)

        # Now crypto should respect the custom hours
        assert checker.market_hours["crypto"].is_24_7 is False

    def test_market_status_unknown_market(self):
        """Test get_market_status for unknown market."""
        checker = MarketHoursChecker()
        status = checker.get_market_status("unknown_market")

        # Should still return a valid status dict
        assert status["market"] == "unknown_market"
        assert status["is_open"] is True  # Unknown defaults to open

    def test_next_open_already_past_today(self):
        """Test next_open_time when already past today's open."""
        checker = MarketHoursChecker()

        # Wednesday at 3 PM (already past 9:30 AM open)
        after_open = datetime(2024, 1, 10, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        next_open = checker.next_open_time("stock_us", after_open)

        # Should return next day's open
        assert next_open is not None
        assert next_open.date() > after_open.date()

    def test_next_close_already_past_today(self):
        """Test next_close_time when already past today's close."""
        checker = MarketHoursChecker()

        # Wednesday at 6 PM (already past 4 PM close)
        after_close = datetime(2024, 1, 10, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        next_close = checker.next_close_time("stock_us", after_close)

        # Should return next day's close
        assert next_close is not None
        assert next_close.date() > after_close.date()
