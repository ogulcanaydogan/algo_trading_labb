"""
Scheduler Utility for AI Summary Notifications.

Tracks when summaries were last sent and manages daily/weekly timing
to prevent duplicate sends and ensure timely notifications.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class SummaryType(Enum):
    """Types of scheduled summaries."""

    DAILY_MARKET = "daily_market"
    WEEKLY_REVIEW = "weekly_review"
    TRADE_EXPLANATION = "trade_explanation"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled summary."""

    summary_type: SummaryType
    enabled: bool = True
    preferred_hour: int = 18  # Default to 6 PM
    preferred_minute: int = 0
    preferred_day_of_week: int = 6  # Sunday for weekly (0=Monday, 6=Sunday)
    min_interval_hours: int = 20  # Minimum hours between sends (for daily)
    min_interval_days: int = 6  # Minimum days between sends (for weekly)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_type": self.summary_type.value,
            "enabled": self.enabled,
            "preferred_hour": self.preferred_hour,
            "preferred_minute": self.preferred_minute,
            "preferred_day_of_week": self.preferred_day_of_week,
            "min_interval_hours": self.min_interval_hours,
            "min_interval_days": self.min_interval_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleConfig":
        data["summary_type"] = SummaryType(data["summary_type"])
        return cls(**data)


@dataclass
class SendRecord:
    """Record of a sent summary."""

    summary_type: SummaryType
    timestamp: datetime
    success: bool
    message_preview: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_type": self.summary_type.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "message_preview": self.message_preview[:100],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SendRecord":
        data["summary_type"] = SummaryType(data["summary_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class SummaryScheduler:
    """
    Manages scheduling of AI-generated summaries.

    Features:
    - Tracks when each summary type was last sent
    - Prevents duplicate sends within configured intervals
    - Supports preferred send times (e.g., 6 PM daily, Sunday weekly)
    - Persists state to disk for crash recovery
    - Provides status information for dashboards

    Usage:
        scheduler = SummaryScheduler()

        # Check if daily summary should be sent
        if scheduler.should_send(SummaryType.DAILY_MARKET):
            summary = analyst.generate_daily_market_summary(...)
            if summary:
                send_to_telegram(summary)
                scheduler.record_send(SummaryType.DAILY_MARKET, success=True)
    """

    DEFAULT_CONFIGS = {
        SummaryType.DAILY_MARKET: ScheduleConfig(
            summary_type=SummaryType.DAILY_MARKET,
            preferred_hour=18,
            preferred_minute=0,
            min_interval_hours=20,
        ),
        SummaryType.WEEKLY_REVIEW: ScheduleConfig(
            summary_type=SummaryType.WEEKLY_REVIEW,
            preferred_hour=10,
            preferred_minute=0,
            preferred_day_of_week=6,  # Sunday
            min_interval_days=6,
        ),
        SummaryType.TRADE_EXPLANATION: ScheduleConfig(
            summary_type=SummaryType.TRADE_EXPLANATION,
            enabled=True,
            min_interval_hours=0,  # No minimum, sent with each trade
        ),
    }

    def __init__(
        self,
        data_dir: str = "data/scheduler",
        configs: Optional[Dict[SummaryType, ScheduleConfig]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / "scheduler_state.json"
        self.history_file = self.data_dir / "send_history.json"

        # Initialize configs
        self.configs = configs or dict(self.DEFAULT_CONFIGS)

        # Load state
        self.last_send: Dict[SummaryType, datetime] = {}
        self.send_history: List[SendRecord] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load scheduler state from disk."""
        # Load last send times
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    for type_str, ts_str in data.get("last_send", {}).items():
                        try:
                            summary_type = SummaryType(type_str)
                            self.last_send[summary_type] = datetime.fromisoformat(ts_str)
                        except (ValueError, KeyError):
                            continue
                    # Load custom configs
                    for cfg_data in data.get("configs", []):
                        try:
                            cfg = ScheduleConfig.from_dict(cfg_data)
                            self.configs[cfg.summary_type] = cfg
                        except (ValueError, KeyError):
                            continue
            except (json.JSONDecodeError, IOError):
                pass

        # Load send history
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.send_history = [SendRecord.from_dict(r) for r in data]
            except (json.JSONDecodeError, IOError):
                pass

    def _save_state(self) -> None:
        """Save scheduler state to disk."""
        state_data = {
            "last_send": {st.value: ts.isoformat() for st, ts in self.last_send.items()},
            "configs": [cfg.to_dict() for cfg in self.configs.values()],
        }
        with open(self.state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def _save_history(self) -> None:
        """Save send history to disk."""
        # Keep only last 100 records
        records = self.send_history[-100:]
        with open(self.history_file, "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)

    def get_config(self, summary_type: SummaryType) -> ScheduleConfig:
        """Get configuration for a summary type."""
        return self.configs.get(summary_type, self.DEFAULT_CONFIGS[summary_type])

    def update_config(self, config: ScheduleConfig) -> None:
        """Update configuration for a summary type."""
        self.configs[config.summary_type] = config
        self._save_state()

    def should_send(
        self,
        summary_type: SummaryType,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Check if a summary should be sent now.

        Args:
            summary_type: Type of summary to check
            now: Current time (defaults to now)

        Returns:
            True if summary should be sent
        """
        now = now or datetime.now()
        config = self.get_config(summary_type)

        # Check if enabled
        if not config.enabled:
            return False

        # Check last send time
        last_send = self.last_send.get(summary_type)

        if summary_type == SummaryType.DAILY_MARKET:
            return self._should_send_daily(config, last_send, now)
        elif summary_type == SummaryType.WEEKLY_REVIEW:
            return self._should_send_weekly(config, last_send, now)
        elif summary_type == SummaryType.TRADE_EXPLANATION:
            # Trade explanations can always be sent (per-trade)
            return True

        return False

    def _should_send_daily(
        self,
        config: ScheduleConfig,
        last_send: Optional[datetime],
        now: datetime,
    ) -> bool:
        """Check if daily summary should be sent."""
        # If never sent, check if we're past the preferred time today
        if last_send is None:
            preferred_time = now.replace(
                hour=config.preferred_hour,
                minute=config.preferred_minute,
                second=0,
                microsecond=0,
            )
            return now >= preferred_time

        # Check minimum interval
        hours_since = (now - last_send).total_seconds() / 3600
        if hours_since < config.min_interval_hours:
            return False

        # Check if we're past the preferred time today and haven't sent today
        today_preferred = now.replace(
            hour=config.preferred_hour,
            minute=config.preferred_minute,
            second=0,
            microsecond=0,
        )

        # If last send was yesterday (or earlier) and we're past preferred time
        if last_send.date() < now.date() and now >= today_preferred:
            return True

        return False

    def _should_send_weekly(
        self,
        config: ScheduleConfig,
        last_send: Optional[datetime],
        now: datetime,
    ) -> bool:
        """Check if weekly summary should be sent."""
        # Check if today is the preferred day
        if now.weekday() != config.preferred_day_of_week:
            return False

        # If never sent, check if we're past the preferred time
        if last_send is None:
            preferred_time = now.replace(
                hour=config.preferred_hour,
                minute=config.preferred_minute,
                second=0,
                microsecond=0,
            )
            return now >= preferred_time

        # Check minimum interval
        days_since = (now.date() - last_send.date()).days
        if days_since < config.min_interval_days:
            return False

        # Check if we're past the preferred time today
        today_preferred = now.replace(
            hour=config.preferred_hour,
            minute=config.preferred_minute,
            second=0,
            microsecond=0,
        )

        return now >= today_preferred

    def record_send(
        self,
        summary_type: SummaryType,
        success: bool,
        message_preview: str = "",
        error: str = "",
    ) -> SendRecord:
        """
        Record that a summary was sent (or attempted).

        Args:
            summary_type: Type of summary sent
            success: Whether send was successful
            message_preview: Preview of message sent
            error: Error message if failed

        Returns:
            The created SendRecord
        """
        now = datetime.now()

        record = SendRecord(
            summary_type=summary_type,
            timestamp=now,
            success=success,
            message_preview=message_preview,
            error=error,
        )

        self.send_history.append(record)

        if success:
            self.last_send[summary_type] = now

        self._save_state()
        self._save_history()

        return record

    def force_send(self, summary_type: SummaryType) -> None:
        """
        Force next check to return True for this summary type.
        Useful for manual triggering.
        """
        # Remove from last_send to allow immediate send
        self.last_send.pop(summary_type, None)
        self._save_state()

    def get_next_send_time(
        self,
        summary_type: SummaryType,
        now: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """
        Get the next scheduled send time for a summary type.

        Args:
            summary_type: Type of summary
            now: Current time (defaults to now)

        Returns:
            Next scheduled datetime or None if disabled
        """
        now = now or datetime.now()
        config = self.get_config(summary_type)

        if not config.enabled:
            return None

        if summary_type == SummaryType.DAILY_MARKET:
            # Next preferred time
            next_time = now.replace(
                hour=config.preferred_hour,
                minute=config.preferred_minute,
                second=0,
                microsecond=0,
            )
            if now >= next_time:
                next_time += timedelta(days=1)
            return next_time

        elif summary_type == SummaryType.WEEKLY_REVIEW:
            # Find next preferred day
            days_ahead = config.preferred_day_of_week - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            elif days_ahead == 0:
                # Same day, check if past time
                preferred_time = now.replace(
                    hour=config.preferred_hour,
                    minute=config.preferred_minute,
                    second=0,
                    microsecond=0,
                )
                if now >= preferred_time:
                    days_ahead = 7

            next_date = now + timedelta(days=days_ahead)
            return next_date.replace(
                hour=config.preferred_hour,
                minute=config.preferred_minute,
                second=0,
                microsecond=0,
            )

        return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status for dashboard display.

        Returns:
            Status dict with last send times, next scheduled times, etc.
        """
        now = datetime.now()
        status = {
            "summaries": {},
            "total_sends_today": 0,
            "last_error": None,
        }

        for summary_type in SummaryType:
            config = self.get_config(summary_type)
            last_send = self.last_send.get(summary_type)

            summary_status = {
                "enabled": config.enabled,
                "last_send": last_send.isoformat() if last_send else None,
                "next_scheduled": None,
                "should_send_now": self.should_send(summary_type, now),
            }

            next_time = self.get_next_send_time(summary_type, now)
            if next_time:
                summary_status["next_scheduled"] = next_time.isoformat()

            status["summaries"][summary_type.value] = summary_status

        # Count today's sends
        today = date.today()
        status["total_sends_today"] = sum(
            1 for r in self.send_history if r.timestamp.date() == today and r.success
        )

        # Get last error
        errors = [r for r in self.send_history if not r.success]
        if errors:
            last_error = errors[-1]
            status["last_error"] = {
                "type": last_error.summary_type.value,
                "timestamp": last_error.timestamp.isoformat(),
                "error": last_error.error,
            }

        return status

    def get_send_history(
        self,
        summary_type: Optional[SummaryType] = None,
        limit: int = 20,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get send history, optionally filtered.

        Args:
            summary_type: Filter by summary type
            limit: Maximum records to return
            success_only: Only return successful sends

        Returns:
            List of send records as dicts
        """
        records = self.send_history

        if summary_type:
            records = [r for r in records if r.summary_type == summary_type]

        if success_only:
            records = [r for r in records if r.success]

        return [r.to_dict() for r in records[-limit:]]

    def clear_history(self, before_date: Optional[date] = None) -> int:
        """
        Clear send history.

        Args:
            before_date: Only clear records before this date (None = clear all)

        Returns:
            Number of records cleared
        """
        if before_date is None:
            count = len(self.send_history)
            self.send_history = []
        else:
            original_count = len(self.send_history)
            self.send_history = [r for r in self.send_history if r.timestamp.date() >= before_date]
            count = original_count - len(self.send_history)

        self._save_history()
        return count


class ScheduledSummaryRunner:
    """
    Convenience class to run scheduled summaries with callback functions.

    Usage:
        from bot.llm.claude import MarketAnalyst, SummaryScheduler

        analyst = MarketAnalyst()
        scheduler = SummaryScheduler()

        runner = ScheduledSummaryRunner(scheduler)
        runner.register_callback(
            SummaryType.DAILY_MARKET,
            lambda: send_telegram(analyst.generate_daily_market_summary(...))
        )

        # Call periodically (e.g., every hour)
        runner.check_and_run()
    """

    def __init__(self, scheduler: SummaryScheduler):
        self.scheduler = scheduler
        self.callbacks: Dict[SummaryType, Callable[[], Optional[str]]] = {}

    def register_callback(
        self,
        summary_type: SummaryType,
        callback: Callable[[], Optional[str]],
    ) -> None:
        """
        Register a callback for a summary type.

        The callback should return the message sent (for recording)
        or None if it failed.
        """
        self.callbacks[summary_type] = callback

    def check_and_run(self) -> Dict[SummaryType, bool]:
        """
        Check all registered summaries and run callbacks if due.

        Returns:
            Dict of summary_type -> whether it was run successfully
        """
        results = {}

        for summary_type, callback in self.callbacks.items():
            if self.scheduler.should_send(summary_type):
                try:
                    result = callback()
                    success = result is not None
                    self.scheduler.record_send(
                        summary_type=summary_type,
                        success=success,
                        message_preview=result[:100] if result else "",
                    )
                    results[summary_type] = success
                except Exception as e:
                    self.scheduler.record_send(
                        summary_type=summary_type,
                        success=False,
                        error=str(e),
                    )
                    results[summary_type] = False

        return results

    def run_now(self, summary_type: SummaryType) -> bool:
        """
        Force run a specific summary now, regardless of schedule.

        Returns:
            True if successful
        """
        if summary_type not in self.callbacks:
            return False

        callback = self.callbacks[summary_type]

        try:
            result = callback()
            success = result is not None
            self.scheduler.record_send(
                summary_type=summary_type,
                success=success,
                message_preview=result[:100] if result else "",
            )
            return success
        except Exception as e:
            self.scheduler.record_send(
                summary_type=summary_type,
                success=False,
                error=str(e),
            )
            return False
