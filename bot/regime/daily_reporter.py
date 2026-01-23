"""
Daily Reporter - Scheduled Telegram reports.

Features:
- Daily summary at configured time
- Weekly summary on Sundays
- Configurable report schedules
- Performance alerts
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportSchedule:
    """Schedule configuration for a report."""

    name: str
    report_type: str  # "daily", "weekly", "monthly"
    time: time = time(20, 0)  # Default 8 PM
    day_of_week: int = 6  # Sunday for weekly (0=Monday, 6=Sunday)
    enabled: bool = True


@dataclass
class ReporterConfig:
    """Configuration for daily reporter."""

    # Report schedules
    schedules: List[ReportSchedule] = field(
        default_factory=lambda: [
            ReportSchedule(name="daily_summary", report_type="daily", time=time(20, 0)),
            ReportSchedule(
                name="weekly_summary", report_type="weekly", time=time(18, 0), day_of_week=6
            ),
        ]
    )

    # Alert thresholds
    alert_on_large_win: float = 500.0  # Alert if trade wins > $500
    alert_on_large_loss: float = 200.0  # Alert if trade loses > $200
    alert_on_drawdown: float = 0.05  # Alert at 5% drawdown


class DailyReporter:
    """
    Scheduled reporter for daily/weekly trading summaries.
    """

    def __init__(
        self,
        config: Optional[ReporterConfig] = None,
        send_callback: Optional[Callable[[str], bool]] = None,
    ):
        self.config = config or ReporterConfig()
        self._send_callback = send_callback

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._last_reports: Dict[str, datetime] = {}

    def set_send_callback(self, callback: Callable[[str], bool]):
        """Set the callback for sending messages."""
        self._send_callback = callback

    async def start(self):
        """Start the reporter scheduler."""
        if self._running:
            return

        self._running = True
        logger.info("Starting daily reporter")

        # Start scheduler task
        task = asyncio.create_task(self._scheduler_loop())
        self._tasks.append(task)

    async def stop(self):
        """Stop the reporter scheduler."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Daily reporter stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()

            for schedule in self.config.schedules:
                if not schedule.enabled:
                    continue

                if self._should_run(schedule, now):
                    await self._run_report(schedule)

            # Sleep for 1 minute
            await asyncio.sleep(60)

    def _should_run(self, schedule: ReportSchedule, now: datetime) -> bool:
        """Check if a report should run now."""
        # Check if already run today
        last_run = self._last_reports.get(schedule.name)
        if last_run and last_run.date() == now.date():
            return False

        # Check time (within 5 minute window)
        scheduled_time = now.replace(
            hour=schedule.time.hour,
            minute=schedule.time.minute,
            second=0,
            microsecond=0,
        )

        time_diff = abs((now - scheduled_time).total_seconds())
        if time_diff > 300:  # 5 minute window
            return False

        # Check day of week for weekly reports
        if schedule.report_type == "weekly":
            if now.weekday() != schedule.day_of_week:
                return False

        # Check day of month for monthly reports
        if schedule.report_type == "monthly":
            if now.day != 1:  # First of month
                return False

        return True

    async def _run_report(self, schedule: ReportSchedule):
        """Run a scheduled report."""
        logger.info(f"Running {schedule.name} report")

        try:
            # Generate report
            message = self._generate_report(schedule.report_type)

            # Send via callback
            if self._send_callback and message:
                success = self._send_callback(message)
                if success:
                    self._last_reports[schedule.name] = datetime.now()
                    logger.info(f"Sent {schedule.name} report")
                else:
                    logger.error(f"Failed to send {schedule.name} report")
            else:
                logger.warning(f"No send callback configured for {schedule.name}")

        except Exception as e:
            logger.error(f"Error running {schedule.name} report: {e}")

    def _generate_report(self, report_type: str) -> str:
        """Generate report message."""
        try:
            from .performance_tracker import get_performance_tracker

            tracker = get_performance_tracker()
            return tracker.generate_telegram_summary(report_type)
        except Exception as e:
            logger.error(f"Error generating {report_type} report: {e}")
            return ""

    def send_alert(self, message: str):
        """Send an immediate alert."""
        if self._send_callback:
            self._send_callback(message)

    def send_trade_alert(
        self,
        symbol: str,
        side: str,
        pnl: float,
        regime: str = "",
    ):
        """Send alert for significant trade."""
        if pnl > self.config.alert_on_large_win:
            msg = f"""
🎉 *Big Win Alert!*

{symbol} {side.upper()}
P&L: +${pnl:,.2f}
Regime: {regime}
            """
            self.send_alert(msg.strip())

        elif pnl < -self.config.alert_on_large_loss:
            msg = f"""
⚠️ *Loss Alert*

{symbol} {side.upper()}
P&L: -${abs(pnl):,.2f}
Regime: {regime}
            """
            self.send_alert(msg.strip())

    def send_drawdown_alert(self, drawdown: float, equity: float):
        """Send drawdown alert."""
        if drawdown >= self.config.alert_on_drawdown:
            msg = f"""
🚨 *Drawdown Alert*

Current Drawdown: {drawdown:.1%}
Equity: ${equity:,.2f}

Consider reducing position sizes.
            """
            self.send_alert(msg.strip())

    async def send_report_now(self, report_type: str = "daily"):
        """Send a report immediately."""
        message = self._generate_report(report_type)
        if self._send_callback and message:
            return self._send_callback(message)
        return False

    def get_status(self) -> Dict:
        """Get reporter status."""
        return {
            "running": self._running,
            "schedules": [
                {
                    "name": s.name,
                    "type": s.report_type,
                    "time": s.time.strftime("%H:%M"),
                    "enabled": s.enabled,
                    "last_run": self._last_reports.get(s.name, "never"),
                }
                for s in self.config.schedules
            ],
        }


# Global reporter instance
_reporter: Optional[DailyReporter] = None


def get_daily_reporter() -> DailyReporter:
    """Get or create the global daily reporter."""
    global _reporter
    if _reporter is None:
        _reporter = DailyReporter()
    return _reporter


async def setup_daily_reporter(send_func: Callable[[str], bool]) -> DailyReporter:
    """Setup and start the daily reporter."""
    reporter = get_daily_reporter()
    reporter.set_send_callback(send_func)
    await reporter.start()
    return reporter
