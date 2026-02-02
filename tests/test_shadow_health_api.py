"""
Tests for Shadow Health API Endpoint and Metrics Calculator.

Tests cover:
1. Correct parsing of latest daily report
2. Correct rolling 7d aggregation
3. heartbeat_recent logic
4. API endpoint response schema
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.shadow_health_metrics import (
    ShadowHealthMetrics,
    ShadowHealthMetricsCalculator,
    get_shadow_health_metrics,
    DATA_MODE_PAPER_LIVE,
    DATA_MODE_TEST,
    GATE_1_REQUIRED_WEEKS,
    HEARTBEAT_MAX_AGE_HOURS,
)


class TestDailyReportParsing:
    """Tests for correct parsing of latest daily report."""

    def test_parse_latest_daily_report_basic(self, tmp_path):
        """Test parsing a basic daily report."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        # Create a daily report
        report_data = {
            "date": "2026-01-29",
            "timestamp": "2026-01-29T10:30:00",
            "shadow_collection": {
                "logging_healthy": True,
                "decisions_today": 25,
                "decisions_by_mode": {
                    "TEST": 10,
                    "PAPER_LIVE": 15
                },
                "pending_decisions": 0,
                "total_all_time": 500,
            },
            "summary": {
                "overall_health": "HEALTHY",
                "alerts": [],
                "recommendations": [],
            },
        }

        report_path = reports_dir / "daily_shadow_health_2026-01-29.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        latest = calculator._get_latest_daily_report()

        assert latest is not None
        assert latest["date"] == "2026-01-29"
        assert latest["timestamp"] == "2026-01-29T10:30:00"
        assert latest["shadow_collection"]["decisions_by_mode"]["PAPER_LIVE"] == 15

    def test_parse_latest_daily_report_multiple_files(self, tmp_path):
        """Test that most recent report is returned when multiple exist."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        # Create multiple daily reports
        for date in ["2026-01-27", "2026-01-28", "2026-01-29"]:
            report_data = {
                "date": date,
                "timestamp": f"{date}T10:30:00",
                "shadow_collection": {
                    "decisions_by_mode": {"PAPER_LIVE": int(date[-2:])},
                },
                "summary": {"overall_health": "HEALTHY"},
            }
            report_path = reports_dir / f"daily_shadow_health_{date}.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        latest = calculator._get_latest_daily_report()

        assert latest is not None
        assert latest["date"] == "2026-01-29"  # Most recent

    def test_parse_latest_daily_report_no_files(self, tmp_path):
        """Test that None is returned when no reports exist."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        latest = calculator._get_latest_daily_report()

        assert latest is None

    def test_parse_latest_daily_report_invalid_json(self, tmp_path):
        """Test that invalid JSON is handled gracefully."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        report_path = reports_dir / "daily_shadow_health_2026-01-29.json"
        with open(report_path, "w") as f:
            f.write("not valid json {{{")

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        latest = calculator._get_latest_daily_report()

        assert latest is None

    def test_paper_live_decisions_today_from_report(self, tmp_path):
        """Test extracting PAPER_LIVE decisions today from report."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        report_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "shadow_collection": {
                "decisions_by_mode": {
                    "TEST": 5,
                    "PAPER_LIVE": 20
                },
            },
            "summary": {"overall_health": "HEALTHY"},
        }

        report_path = reports_dir / f"daily_shadow_health_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        metrics = calculator.calculate_metrics()

        assert metrics.paper_live_decisions_today == 20


class TestRolling7dAggregation:
    """Tests for correct rolling 7-day aggregation."""

    def test_7d_aggregation_basic(self, tmp_path):
        """Test 7-day rolling aggregation with reports in range."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        # Create 7 days of reports
        base_date = datetime.now()
        for i in range(7):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            report_data = {
                "date": date,
                "timestamp": f"{date}T10:30:00",
                "shadow_collection": {
                    "decisions_by_mode": {
                        "TEST": 5,
                        "PAPER_LIVE": 10  # 10 PAPER_LIVE per day
                    },
                },
                "summary": {"overall_health": "HEALTHY"},
            }
            report_path = reports_dir / f"daily_shadow_health_{date}.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        total_7d = calculator._calculate_7d_paper_live()

        assert total_7d == 70  # 10 * 7 days

    def test_7d_aggregation_excludes_old_reports(self, tmp_path):
        """Test that reports older than 7 days are excluded."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        base_date = datetime.now()

        # Create a report from 10 days ago (should be excluded)
        old_date = (base_date - timedelta(days=10)).strftime("%Y-%m-%d")
        old_report = {
            "date": old_date,
            "shadow_collection": {"decisions_by_mode": {"PAPER_LIVE": 100}},
            "summary": {"overall_health": "HEALTHY"},
        }
        with open(reports_dir / f"daily_shadow_health_{old_date}.json", "w") as f:
            json.dump(old_report, f)

        # Create a report from today (should be included)
        today = base_date.strftime("%Y-%m-%d")
        today_report = {
            "date": today,
            "shadow_collection": {"decisions_by_mode": {"PAPER_LIVE": 25}},
            "summary": {"overall_health": "HEALTHY"},
        }
        with open(reports_dir / f"daily_shadow_health_{today}.json", "w") as f:
            json.dump(today_report, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        total_7d = calculator._calculate_7d_paper_live()

        assert total_7d == 25  # Only today's report

    def test_7d_aggregation_counts_only_paper_live(self, tmp_path):
        """Test that only PAPER_LIVE decisions are counted, not TEST."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        today = datetime.now().strftime("%Y-%m-%d")
        report_data = {
            "date": today,
            "shadow_collection": {
                "decisions_by_mode": {
                    "TEST": 50,
                    "PAPER_LIVE": 15
                },
            },
            "summary": {"overall_health": "HEALTHY"},
        }
        with open(reports_dir / f"daily_shadow_health_{today}.json", "w") as f:
            json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        total_7d = calculator._calculate_7d_paper_live()

        assert total_7d == 15  # Only PAPER_LIVE, not 50 TEST

    def test_7d_aggregation_empty_reports_dir(self, tmp_path):
        """Test 7d aggregation returns 0 when no reports exist."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        total_7d = calculator._calculate_7d_paper_live()

        assert total_7d == 0


class TestHeartbeatRecentLogic:
    """Tests for heartbeat_recent logic."""

    def test_heartbeat_recent_true(self, tmp_path):
        """Test heartbeat_recent=1 when heartbeat is fresh."""
        heartbeat_path = tmp_path / "heartbeat.json"

        # Fresh heartbeat (just now)
        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "pid": 12345,
            "mode": "PAPER_LIVE",
            "shadow_collector_attached": True,
        }
        with open(heartbeat_path, "w") as f:
            json.dump(heartbeat_data, f)

        calculator = ShadowHealthMetricsCalculator(heartbeat_path=heartbeat_path)
        recent = calculator._check_heartbeat_recent()

        assert recent == 1

    def test_heartbeat_recent_false_stale(self, tmp_path):
        """Test heartbeat_recent=0 when heartbeat is stale."""
        heartbeat_path = tmp_path / "heartbeat.json"

        # 3-hour old heartbeat (> 2 hour threshold)
        stale_time = (datetime.now() - timedelta(hours=3)).isoformat()
        heartbeat_data = {
            "timestamp": stale_time,
            "pid": 12345,
            "mode": "PAPER_LIVE",
        }
        with open(heartbeat_path, "w") as f:
            json.dump(heartbeat_data, f)

        calculator = ShadowHealthMetricsCalculator(heartbeat_path=heartbeat_path)
        recent = calculator._check_heartbeat_recent()

        assert recent == 0

    def test_heartbeat_recent_false_missing(self, tmp_path):
        """Test heartbeat_recent=0 when heartbeat file doesn't exist."""
        heartbeat_path = tmp_path / "nonexistent_heartbeat.json"

        calculator = ShadowHealthMetricsCalculator(heartbeat_path=heartbeat_path)
        recent = calculator._check_heartbeat_recent()

        assert recent == 0

    def test_heartbeat_recent_boundary(self, tmp_path):
        """Test heartbeat at exactly the boundary (1.9 hours)."""
        heartbeat_path = tmp_path / "heartbeat.json"

        # Just under 2 hours old
        boundary_time = (datetime.now() - timedelta(hours=1.9)).isoformat()
        heartbeat_data = {
            "timestamp": boundary_time,
            "pid": 12345,
        }
        with open(heartbeat_path, "w") as f:
            json.dump(heartbeat_data, f)

        calculator = ShadowHealthMetricsCalculator(heartbeat_path=heartbeat_path)
        recent = calculator._check_heartbeat_recent()

        assert recent == 1  # Still recent (< 2 hours)

    def test_heartbeat_recent_invalid_json(self, tmp_path):
        """Test heartbeat_recent=0 when file contains invalid JSON."""
        heartbeat_path = tmp_path / "heartbeat.json"

        with open(heartbeat_path, "w") as f:
            f.write("invalid json")

        calculator = ShadowHealthMetricsCalculator(heartbeat_path=heartbeat_path)
        recent = calculator._check_heartbeat_recent()

        assert recent == 0

    def test_heartbeat_recent_missing_timestamp(self, tmp_path):
        """Test heartbeat_recent=0 when timestamp field is missing."""
        heartbeat_path = tmp_path / "heartbeat.json"

        heartbeat_data = {"pid": 12345, "mode": "PAPER_LIVE"}  # No timestamp
        with open(heartbeat_path, "w") as f:
            json.dump(heartbeat_data, f)

        calculator = ShadowHealthMetricsCalculator(heartbeat_path=heartbeat_path)
        recent = calculator._check_heartbeat_recent()

        assert recent == 0


class TestPaperLiveStreak:
    """Tests for consecutive days streak calculation."""

    def test_streak_consecutive_days(self, tmp_path):
        """Test streak calculation with consecutive PAPER_LIVE days."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        base_date = datetime.now()
        # Create 5 consecutive days of PAPER_LIVE data
        for i in range(5):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            report_data = {
                "date": date,
                "shadow_collection": {
                    "decisions_by_mode": {"PAPER_LIVE": 10},
                },
                "summary": {"overall_health": "HEALTHY"},
            }
            with open(reports_dir / f"daily_shadow_health_{date}.json", "w") as f:
                json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        streak = calculator._calculate_paper_live_streak()

        assert streak == 5

    def test_streak_broken_by_gap(self, tmp_path):
        """Test streak is broken by missing days."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        base_date = datetime.now()

        # Today and yesterday have PAPER_LIVE
        for i in range(2):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            report_data = {
                "date": date,
                "shadow_collection": {"decisions_by_mode": {"PAPER_LIVE": 10}},
            }
            with open(reports_dir / f"daily_shadow_health_{date}.json", "w") as f:
                json.dump(report_data, f)

        # Skip day 2, then day 3 has PAPER_LIVE (shouldn't count)
        date_3 = (base_date - timedelta(days=3)).strftime("%Y-%m-%d")
        report_data = {
            "date": date_3,
            "shadow_collection": {"decisions_by_mode": {"PAPER_LIVE": 10}},
        }
        with open(reports_dir / f"daily_shadow_health_{date_3}.json", "w") as f:
            json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        streak = calculator._calculate_paper_live_streak()

        assert streak == 2  # Only today and yesterday

    def test_streak_broken_by_zero_paper_live(self, tmp_path):
        """Test streak is broken when a day has 0 PAPER_LIVE."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        base_date = datetime.now()

        # Today has PAPER_LIVE
        today = base_date.strftime("%Y-%m-%d")
        with open(reports_dir / f"daily_shadow_health_{today}.json", "w") as f:
            json.dump({"date": today, "shadow_collection": {"decisions_by_mode": {"PAPER_LIVE": 10}}}, f)

        # Yesterday has 0 PAPER_LIVE (only TEST)
        yesterday = (base_date - timedelta(days=1)).strftime("%Y-%m-%d")
        with open(reports_dir / f"daily_shadow_health_{yesterday}.json", "w") as f:
            json.dump({"date": yesterday, "shadow_collection": {"decisions_by_mode": {"TEST": 20, "PAPER_LIVE": 0}}}, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        streak = calculator._calculate_paper_live_streak()

        assert streak == 1  # Only today counts


class TestPaperLiveWeeksCounted:
    """Tests for Gate 1 (weeks counted) calculation."""

    def test_count_paper_live_weeks(self, tmp_path):
        """Test counting weeks with PAPER_LIVE data."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        # Create 5 weekly reports: 3 PAPER_LIVE, 2 TEST
        for i in range(5):
            mode = "PAPER_LIVE" if i < 3 else "TEST"
            report_data = {
                "week_ending": f"2026-01-{15 + i * 7:02d}",
                "data_mode": mode,
            }
            with open(reports_dir / f"weekly_shadow_report_2026-01-{15 + i * 7:02d}.json", "w") as f:
                json.dump(report_data, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        weeks = calculator._count_paper_live_weeks()

        assert weeks == 3

    def test_count_weeks_backwards_compatibility(self, tmp_path):
        """Test that missing data_mode defaults to TEST."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        # Old format without data_mode
        old_report = {"week_ending": "2026-01-15"}  # No data_mode field
        with open(reports_dir / "weekly_shadow_report_2026-01-15.json", "w") as f:
            json.dump(old_report, f)

        calculator = ShadowHealthMetricsCalculator(reports_dir=reports_dir)
        weeks = calculator._count_paper_live_weeks()

        assert weeks == 0  # Defaults to TEST, not counted


class TestFullMetricsCalculation:
    """Tests for complete metrics calculation."""

    def test_full_metrics_calculation(self, tmp_path):
        """Test complete metrics calculation with all components."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        heartbeat_path = tmp_path / "heartbeat.json"

        # Create daily report for today
        today = datetime.now().strftime("%Y-%m-%d")
        daily_report = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "shadow_collection": {
                "decisions_by_mode": {"PAPER_LIVE": 25, "TEST": 5},
            },
            "summary": {"overall_health": "HEALTHY"},
        }
        with open(reports_dir / f"daily_shadow_health_{today}.json", "w") as f:
            json.dump(daily_report, f)

        # Create weekly reports (2 PAPER_LIVE)
        for i in range(2):
            weekly_report = {"data_mode": "PAPER_LIVE"}
            with open(reports_dir / f"weekly_shadow_report_week_{i}.json", "w") as f:
                json.dump(weekly_report, f)

        # Create fresh heartbeat
        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "pid": 12345,
        }
        with open(heartbeat_path, "w") as f:
            json.dump(heartbeat_data, f)

        calculator = ShadowHealthMetricsCalculator(
            reports_dir=reports_dir,
            heartbeat_path=heartbeat_path,
        )
        metrics = calculator.calculate_metrics()

        assert metrics.paper_live_decisions_today == 25
        assert metrics.paper_live_decisions_7d == 25
        assert metrics.paper_live_days_streak == 1
        assert metrics.paper_live_weeks_counted == 2
        assert metrics.heartbeat_recent == 1
        assert metrics.overall_health == "HEALTHY"
        assert metrics.gate_1_progress["required"] == 12
        assert metrics.gate_1_progress["current"] == 2
        assert metrics.gate_1_progress["met"] is False

    def test_metrics_to_dict(self, tmp_path):
        """Test that metrics serialize correctly to dict."""
        metrics = ShadowHealthMetrics(
            paper_live_decisions_today=10,
            paper_live_decisions_7d=50,
            paper_live_days_streak=3,
            paper_live_weeks_counted=5,
            heartbeat_recent=1,
            latest_report_timestamp="2026-01-29T10:00:00",
            overall_health="HEALTHY",
            gate_1_progress={"required": 12, "current": 5, "met": False},
        )

        d = metrics.to_dict()

        assert d["paper_live_decisions_today"] == 10
        assert d["paper_live_decisions_7d"] == 50
        assert d["paper_live_days_streak"] == 3
        assert d["paper_live_weeks_counted"] == 5
        assert d["heartbeat_recent"] == 1
        assert d["latest_report_timestamp"] == "2026-01-29T10:00:00"
        assert d["overall_health"] == "HEALTHY"
        assert d["gate_1_progress"]["met"] is False


class TestAPIResponseSchema:
    """Tests for API response schema validation."""

    def test_shadow_health_response_schema(self):
        """Test that ShadowHealthResponse has all required fields."""
        from api.schemas import ShadowHealthResponse

        # Create a valid response
        response = ShadowHealthResponse(
            paper_live_decisions_today=10,
            paper_live_decisions_7d=50,
            paper_live_days_streak=3,
            paper_live_weeks_counted=5,
            heartbeat_recent=1,
            latest_report_timestamp="2026-01-29T10:00:00",
            gate_1_progress={"required": 12, "current": 5, "met": False},
            overall_health="HEALTHY",
        )

        # Verify all fields are accessible
        assert response.paper_live_decisions_today == 10
        assert response.paper_live_decisions_7d == 50
        assert response.paper_live_days_streak == 3
        assert response.paper_live_weeks_counted == 5
        assert response.heartbeat_recent == 1
        assert response.latest_report_timestamp == "2026-01-29T10:00:00"
        assert response.gate_1_progress["met"] is False
        assert response.overall_health == "HEALTHY"

    def test_shadow_health_response_optional_timestamp(self):
        """Test that latest_report_timestamp can be None."""
        from api.schemas import ShadowHealthResponse

        response = ShadowHealthResponse(
            paper_live_decisions_today=0,
            paper_live_decisions_7d=0,
            paper_live_days_streak=0,
            paper_live_weeks_counted=0,
            heartbeat_recent=0,
            latest_report_timestamp=None,
            gate_1_progress={"required": 12, "current": 0, "met": False},
            overall_health="UNKNOWN",
        )

        assert response.latest_report_timestamp is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
