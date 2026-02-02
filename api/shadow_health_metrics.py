"""
Shadow Health Metrics Calculator.

Provides PAPER_LIVE progress metrics for dashboard display and API endpoints.
This is a READ-ONLY module with NO execution authority.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants
DATA_MODE_PAPER_LIVE = "PAPER_LIVE"
DATA_MODE_TEST = "TEST"
GATE_1_REQUIRED_WEEKS = 12
HEARTBEAT_MAX_AGE_HOURS = 2.0

# Default paths
DEFAULT_REPORTS_DIR = Path("data/reports")
DEFAULT_SHADOW_LOG = Path("data/rl/shadow_decisions.jsonl")
DEFAULT_HEARTBEAT_PATH = Path("data/rl/paper_live_heartbeat.json")
DEFAULT_WEEKLY_REPORTS_DIR = Path("data/reports")


@dataclass
class ShadowHealthMetrics:
    """Container for shadow health metrics."""

    paper_live_decisions_today: int = 0
    paper_live_decisions_7d: int = 0
    paper_live_days_streak: int = 0
    paper_live_weeks_counted: int = 0
    heartbeat_recent: int = 0  # 0 or 1
    latest_report_timestamp: Optional[str] = None
    overall_health: str = "UNKNOWN"
    gate_1_progress: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_live_decisions_today": self.paper_live_decisions_today,
            "paper_live_decisions_7d": self.paper_live_decisions_7d,
            "paper_live_days_streak": self.paper_live_days_streak,
            "paper_live_weeks_counted": self.paper_live_weeks_counted,
            "heartbeat_recent": self.heartbeat_recent,
            "latest_report_timestamp": self.latest_report_timestamp,
            "gate_1_progress": self.gate_1_progress,
            "overall_health": self.overall_health,
        }


class ShadowHealthMetricsCalculator:
    """Calculates PAPER_LIVE progress metrics from shadow data files."""

    def __init__(
        self,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
        shadow_log_path: Path = DEFAULT_SHADOW_LOG,
        heartbeat_path: Path = DEFAULT_HEARTBEAT_PATH,
    ):
        self.reports_dir = reports_dir
        self.shadow_log_path = shadow_log_path
        self.heartbeat_path = heartbeat_path

    def calculate_metrics(self) -> ShadowHealthMetrics:
        """Calculate all shadow health metrics."""
        metrics = ShadowHealthMetrics()

        # Get latest daily report
        latest_report = self._get_latest_daily_report()
        if latest_report:
            metrics.latest_report_timestamp = latest_report.get("timestamp")
            metrics.overall_health = latest_report.get("summary", {}).get(
                "overall_health", "UNKNOWN"
            )

            # Get today's PAPER_LIVE decisions from latest report
            decisions_by_mode = latest_report.get("shadow_collection", {}).get(
                "decisions_by_mode", {}
            )
            metrics.paper_live_decisions_today = decisions_by_mode.get(
                DATA_MODE_PAPER_LIVE, 0
            )

        # Calculate 7-day rolling aggregation
        metrics.paper_live_decisions_7d = self._calculate_7d_paper_live()

        # Calculate consecutive days streak
        metrics.paper_live_days_streak = self._calculate_paper_live_streak()

        # Count PAPER_LIVE weeks (Gate 1)
        metrics.paper_live_weeks_counted = self._count_paper_live_weeks()

        # Check heartbeat
        metrics.heartbeat_recent = self._check_heartbeat_recent()

        # Gate 1 progress
        metrics.gate_1_progress = {
            "required": GATE_1_REQUIRED_WEEKS,
            "current": metrics.paper_live_weeks_counted,
            "met": metrics.paper_live_weeks_counted >= GATE_1_REQUIRED_WEEKS,
        }

        return metrics

    def _get_latest_daily_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent daily shadow health report."""
        if not self.reports_dir.exists():
            return None

        report_files = sorted(
            self.reports_dir.glob("daily_shadow_health_*.json"),
            key=lambda p: p.name,
            reverse=True,
        )

        if not report_files:
            return None

        try:
            with open(report_files[0]) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read latest daily report: {e}")
            return None

    def _calculate_7d_paper_live(self) -> int:
        """Calculate PAPER_LIVE decisions in the last 7 days."""
        if not self.reports_dir.exists():
            return 0

        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        total = 0

        for report_file in self.reports_dir.glob("daily_shadow_health_*.json"):
            # Extract date from filename: daily_shadow_health_YYYY-MM-DD.json
            try:
                date_str = report_file.stem.replace("daily_shadow_health_", "")
                if date_str >= cutoff_date:
                    with open(report_file) as f:
                        report = json.load(f)
                    decisions_by_mode = report.get("shadow_collection", {}).get(
                        "decisions_by_mode", {}
                    )
                    total += decisions_by_mode.get(DATA_MODE_PAPER_LIVE, 0)
            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to read report {report_file}: {e}")
                continue

        return total

    def _calculate_paper_live_streak(self) -> int:
        """Calculate consecutive days with PAPER_LIVE data."""
        if not self.reports_dir.exists():
            return 0

        # Get all daily reports sorted by date descending
        report_files = sorted(
            self.reports_dir.glob("daily_shadow_health_*.json"),
            key=lambda p: p.name,
            reverse=True,
        )

        streak = 0
        expected_date = datetime.now().date()

        for report_file in report_files:
            try:
                date_str = report_file.stem.replace("daily_shadow_health_", "")
                report_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # Check if this is the expected date in the streak
                if report_date != expected_date:
                    # Gap in dates, streak ends
                    break

                with open(report_file) as f:
                    report = json.load(f)

                decisions_by_mode = report.get("shadow_collection", {}).get(
                    "decisions_by_mode", {}
                )
                paper_live_count = decisions_by_mode.get(DATA_MODE_PAPER_LIVE, 0)

                if paper_live_count > 0:
                    streak += 1
                    expected_date -= timedelta(days=1)
                else:
                    # No PAPER_LIVE data this day, streak ends
                    break

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to read report {report_file}: {e}")
                break

        return streak

    def _count_paper_live_weeks(self) -> int:
        """Count number of weekly reports with PAPER_LIVE data."""
        if not self.reports_dir.exists():
            return 0

        count = 0
        for report_file in self.reports_dir.glob("weekly_shadow_report_*.json"):
            try:
                with open(report_file) as f:
                    report = json.load(f)

                # Check data_mode - default to TEST for backwards compatibility
                data_mode = report.get("data_mode", DATA_MODE_TEST)
                if data_mode == DATA_MODE_PAPER_LIVE:
                    count += 1

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read weekly report {report_file}: {e}")
                continue

        return count

    def _check_heartbeat_recent(self) -> int:
        """Check if paper-live heartbeat is recent. Returns 1 if recent, 0 otherwise."""
        if not self.heartbeat_path.exists():
            return 0

        try:
            with open(self.heartbeat_path) as f:
                heartbeat = json.load(f)

            timestamp_str = heartbeat.get("timestamp")
            if not timestamp_str:
                return 0

            heartbeat_ts = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - heartbeat_ts
            age_hours = age.total_seconds() / 3600

            return 1 if age_hours < HEARTBEAT_MAX_AGE_HOURS else 0

        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Failed to read heartbeat: {e}")
            return 0


def get_shadow_health_metrics(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    shadow_log_path: Path = DEFAULT_SHADOW_LOG,
    heartbeat_path: Path = DEFAULT_HEARTBEAT_PATH,
) -> ShadowHealthMetrics:
    """Convenience function to get shadow health metrics."""
    calculator = ShadowHealthMetricsCalculator(
        reports_dir=reports_dir,
        shadow_log_path=shadow_log_path,
        heartbeat_path=heartbeat_path,
    )
    return calculator.calculate_metrics()
