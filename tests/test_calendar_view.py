"""Tests for bot.calendar_view module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pytest

from bot.calendar_view import (
    DayPerformance,
    MonthPerformance,
    CalendarData,
    CalendarViewGenerator,
)


class TestDayPerformance:
    """Tests for DayPerformance dataclass."""

    def test_creation(self) -> None:
        """Test DayPerformance creation."""
        day = DayPerformance(
            date="2024-01-15",
            pnl=100.0,
            pnl_percent=2.5,
            trades=5,
            wins=3,
            losses=2,
            best_trade=50.0,
            worst_trade=-20.0,
            volume=10000.0,
        )

        assert day.date == "2024-01-15"
        assert day.pnl == 100.0
        assert day.trades == 5
        assert day.wins == 3


class TestMonthPerformance:
    """Tests for MonthPerformance dataclass."""

    def test_creation(self) -> None:
        """Test MonthPerformance creation."""
        month = MonthPerformance(
            year=2024,
            month=1,
            total_pnl=500.0,
            total_pnl_percent=5.0,
            trading_days=20,
            winning_days=12,
            losing_days=8,
            best_day=100.0,
            worst_day=-50.0,
            avg_daily_pnl=25.0,
        )

        assert month.year == 2024
        assert month.month == 1
        assert month.trading_days == 20


class TestCalendarViewGenerator:
    """Tests for CalendarViewGenerator class."""

    @pytest.fixture
    def generator(self) -> CalendarViewGenerator:
        """Create generator instance."""
        return CalendarViewGenerator()

    @pytest.fixture
    def sample_trades(self) -> List[Dict[str, Any]]:
        """Create sample trade data."""
        trades = []
        base_date = datetime(2024, 1, 1)

        for i in range(30):
            trade_date = base_date + timedelta(days=i)
            pnl = 50.0 if i % 3 != 0 else -30.0
            trades.append({
                "timestamp": trade_date.isoformat(),
                "pnl": pnl,
                "pnl_percent": pnl / 1000,  # Assuming $1000 base
                "quantity": 1.0,
                "entry_price": 100.0,
            })

        return trades

    def test_init(self, generator: CalendarViewGenerator) -> None:
        """Test generator initialization."""
        assert generator._daily_data == {}
        assert generator._trades == []

    def test_load_trades(
        self,
        generator: CalendarViewGenerator,
        sample_trades: List[Dict],
    ) -> None:
        """Test loading trades."""
        generator.load_trades(sample_trades)

        assert len(generator._trades) == 30
        assert len(generator._daily_data) == 30

    def test_load_from_json_list(
        self,
        generator: CalendarViewGenerator,
        sample_trades: List[Dict],
        tmp_path: Path,
    ) -> None:
        """Test loading trades from JSON file with list format."""
        json_path = tmp_path / "trades.json"
        with open(json_path, "w") as f:
            json.dump(sample_trades, f)

        generator.load_from_json(str(json_path))

        assert len(generator._trades) == 30

    def test_load_from_json_dict(
        self,
        generator: CalendarViewGenerator,
        sample_trades: List[Dict],
        tmp_path: Path,
    ) -> None:
        """Test loading trades from JSON file with dict format."""
        json_path = tmp_path / "trades.json"
        with open(json_path, "w") as f:
            json.dump({"trades": sample_trades}, f)

        generator.load_from_json(str(json_path))

        assert len(generator._trades) == 30

    def test_load_from_json_nonexistent(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test loading from non-existent file."""
        generator.load_from_json("/nonexistent/path.json")

        assert generator._trades == []

    def test_process_trades_string_timestamp(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test processing trades with string timestamps."""
        trades = [
            {"timestamp": "2024-01-15T10:00:00Z", "pnl": 100, "pnl_percent": 1.0},
            {"timestamp": "2024-01-15T14:00:00Z", "pnl": -50, "pnl_percent": -0.5},
        ]

        generator.load_trades(trades)

        assert "2024-01-15" in generator._daily_data
        day = generator._daily_data["2024-01-15"]
        assert day.pnl == 50  # 100 - 50
        assert day.trades == 2

    def test_process_trades_numeric_timestamp(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test processing trades with numeric timestamps."""
        # Unix timestamp in seconds
        ts = datetime(2024, 1, 15, 10, 0, 0).timestamp()
        trades = [
            {"timestamp": ts, "pnl": 100, "pnl_percent": 1.0},
        ]

        generator.load_trades(trades)

        assert "2024-01-15" in generator._daily_data

    def test_process_trades_millisecond_timestamp(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test processing trades with millisecond timestamps."""
        # Unix timestamp in milliseconds
        ts = datetime(2024, 1, 15, 10, 0, 0).timestamp() * 1000
        trades = [
            {"timestamp": ts, "pnl": 100, "pnl_percent": 1.0},
        ]

        generator.load_trades(trades)

        assert "2024-01-15" in generator._daily_data

    def test_process_trades_close_time_field(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test processing trades with close_time field."""
        trades = [
            {"close_time": "2024-01-15T10:00:00", "pnl": 100, "pnl_percent": 1.0},
        ]

        generator.load_trades(trades)

        assert "2024-01-15" in generator._daily_data

    def test_generate_calendar(
        self,
        generator: CalendarViewGenerator,
        sample_trades: List[Dict],
    ) -> None:
        """Test calendar generation."""
        generator.load_trades(sample_trades)

        calendar = generator.generate_calendar()

        assert isinstance(calendar, CalendarData)
        assert len(calendar.days) == 30
        assert len(calendar.months) >= 1
        assert len(calendar.heatmap_data) == 30
        assert "total_days" in calendar.statistics

    def test_generate_calendar_with_date_filter(
        self,
        generator: CalendarViewGenerator,
        sample_trades: List[Dict],
    ) -> None:
        """Test calendar generation with date filtering."""
        generator.load_trades(sample_trades)

        calendar = generator.generate_calendar(
            start_date=datetime(2024, 1, 10),
            end_date=datetime(2024, 1, 20),
        )

        assert len(calendar.days) == 11  # Jan 10-20 inclusive

    def test_calculate_monthly_summaries(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test monthly summary calculation."""
        days = [
            DayPerformance(
                date="2024-01-15",
                pnl=100,
                pnl_percent=1.0,
                trades=5,
                wins=3,
                losses=2,
                best_trade=50,
                worst_trade=-10,
                volume=1000,
            ),
            DayPerformance(
                date="2024-01-16",
                pnl=-50,
                pnl_percent=-0.5,
                trades=3,
                wins=1,
                losses=2,
                best_trade=20,
                worst_trade=-40,
                volume=500,
            ),
        ]

        months = generator._calculate_monthly_summaries(days)

        assert len(months) == 1
        assert months[0].year == 2024
        assert months[0].month == 1
        assert months[0].total_pnl == 50  # 100 - 50
        assert months[0].trading_days == 2
        assert months[0].winning_days == 1
        assert months[0].losing_days == 1

    def test_calculate_yearly_summaries(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test yearly summary calculation."""
        months = [
            MonthPerformance(
                year=2024,
                month=1,
                total_pnl=100,
                total_pnl_percent=1.0,
                trading_days=20,
                winning_days=12,
                losing_days=8,
                best_day=50,
                worst_day=-30,
                avg_daily_pnl=5,
            ),
            MonthPerformance(
                year=2024,
                month=2,
                total_pnl=150,
                total_pnl_percent=1.5,
                trading_days=18,
                winning_days=10,
                losing_days=8,
                best_day=60,
                worst_day=-25,
                avg_daily_pnl=8.3,
            ),
        ]

        yearly = generator._calculate_yearly_summaries(months)

        assert 2024 in yearly
        assert yearly[2024]["total_pnl"] == 250
        assert yearly[2024]["trading_days"] == 38
        assert yearly[2024]["best_day"] == 60
        assert yearly[2024]["worst_day"] == -30

    def test_generate_heatmap_data(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test heatmap data generation."""
        days = [
            DayPerformance(
                date="2024-01-15",
                pnl=100,
                pnl_percent=5.0,
                trades=5,
                wins=3,
                losses=2,
                best_trade=50,
                worst_trade=-10,
                volume=1000,
            ),
        ]

        heatmap = generator._generate_heatmap_data(days)

        assert len(heatmap) == 1
        assert heatmap[0]["date"] == "2024-01-15"
        assert heatmap[0]["year"] == 2024
        assert heatmap[0]["month"] == 1
        assert heatmap[0]["day"] == 15
        assert "weekday" in heatmap[0]
        assert "intensity" in heatmap[0]
        assert "color" in heatmap[0]

    def test_get_color_positive(self, generator: CalendarViewGenerator) -> None:
        """Test color for positive intensity."""
        assert generator._get_color(0.9) == "#1B5E20"  # Dark green
        assert generator._get_color(0.6) == "#2E7D32"
        assert generator._get_color(0.3) == "#43A047"
        assert generator._get_color(0.1) == "#66BB6A"

    def test_get_color_neutral(self, generator: CalendarViewGenerator) -> None:
        """Test color for neutral intensity."""
        assert generator._get_color(0.0) == "#9E9E9E"  # Gray

    def test_get_color_negative(self, generator: CalendarViewGenerator) -> None:
        """Test color for negative intensity."""
        assert generator._get_color(-0.1) == "#EF5350"
        assert generator._get_color(-0.3) == "#E53935"
        assert generator._get_color(-0.6) == "#C62828"
        assert generator._get_color(-0.9) == "#B71C1C"  # Dark red

    def test_calculate_statistics_empty(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test statistics for empty data."""
        stats = generator._calculate_statistics([])

        assert stats["total_days"] == 0
        assert stats["win_rate"] == 0

    def test_calculate_statistics(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test statistics calculation."""
        days = [
            DayPerformance(
                date=f"2024-01-{15+i:02d}",
                pnl=100 if i % 2 == 0 else -50,
                pnl_percent=1.0 if i % 2 == 0 else -0.5,
                trades=5,
                wins=3 if i % 2 == 0 else 1,
                losses=2 if i % 2 == 0 else 4,
                best_trade=50,
                worst_trade=-10,
                volume=1000,
            )
            for i in range(10)
        ]

        stats = generator._calculate_statistics(days)

        assert stats["total_days"] == 10
        assert stats["winning_days"] == 5  # Even indices
        assert stats["losing_days"] == 5  # Odd indices
        assert stats["win_rate"] == 50.0
        assert "best_day" in stats
        assert "worst_day" in stats
        assert "best_streak" in stats
        assert "worst_streak" in stats
        assert "weekday_performance" in stats

    def test_calculate_streaks_winning(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test streak calculation with winning streak."""
        days = [
            DayPerformance(
                date=f"2024-01-{i:02d}",
                pnl=100 if i <= 5 else -50,
                pnl_percent=1.0,
                trades=1,
                wins=1,
                losses=0,
                best_trade=100,
                worst_trade=0,
                volume=1000,
            )
            for i in range(1, 11)
        ]

        best_streak, worst_streak = generator._calculate_streaks(days)

        assert best_streak == 5  # Days 1-5
        assert worst_streak == 5  # Days 6-10 (5 consecutive losing days)

    def test_calculate_streaks_neutral_days(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test streak calculation with neutral (zero pnl) days."""
        days = [
            DayPerformance(
                date="2024-01-01", pnl=100, pnl_percent=1.0,
                trades=1, wins=1, losses=0, best_trade=100, worst_trade=0, volume=1000,
            ),
            DayPerformance(
                date="2024-01-02", pnl=0, pnl_percent=0,  # Neutral
                trades=0, wins=0, losses=0, best_trade=0, worst_trade=0, volume=0,
            ),
            DayPerformance(
                date="2024-01-03", pnl=100, pnl_percent=1.0,
                trades=1, wins=1, losses=0, best_trade=100, worst_trade=0, volume=1000,
            ),
        ]

        best_streak, worst_streak = generator._calculate_streaks(days)

        assert best_streak == 1  # Streak broken by neutral day
        assert worst_streak == 0

    def test_get_month_grid(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test month grid generation."""
        # Add some data for January 2024
        generator._daily_data["2024-01-15"] = DayPerformance(
            date="2024-01-15",
            pnl=100,
            pnl_percent=1.0,
            trades=5,
            wins=3,
            losses=2,
            best_trade=50,
            worst_trade=-10,
            volume=1000,
        )

        grid = generator.get_month_grid(2024, 1)

        # Should have 5-6 weeks
        assert 4 <= len(grid) <= 6
        # Each week should have 7 days
        for week in grid:
            assert len(week) == 7

        # Find the day with data
        found = False
        for week in grid:
            for day in week:
                if day and day.get("date") == "2024-01-15":
                    found = True
                    assert day["pnl"] == 100
                    assert day["trades"] == 5
        assert found

    def test_to_api_response(
        self,
        generator: CalendarViewGenerator,
        sample_trades: List[Dict],
    ) -> None:
        """Test API response format."""
        generator.load_trades(sample_trades)
        calendar = generator.generate_calendar()

        response = generator.to_api_response(calendar)

        assert "statistics" in response
        assert "heatmap" in response
        assert "months" in response
        assert "yearly" in response

        # Check months format
        for month in response["months"]:
            assert "year" in month
            assert "month" in month
            assert "win_rate" in month

    def test_invalid_timestamp_handling(
        self,
        generator: CalendarViewGenerator,
    ) -> None:
        """Test handling of invalid timestamps."""
        trades = [
            {"timestamp": "invalid-date", "pnl": 100},
            {"timestamp": "2024-01-15T10:00:00", "pnl": 50},
            {"pnl": 25},  # No timestamp
        ]

        generator.load_trades(trades)

        # Only the valid trade should be processed
        assert len(generator._daily_data) == 1
        assert "2024-01-15" in generator._daily_data
