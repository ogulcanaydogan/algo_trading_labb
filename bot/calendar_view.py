"""
Backtest Calendar View Module.

Provides calendar heatmap visualization of daily returns
for performance analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DayPerformance:
    """Performance metrics for a single day."""
    date: str
    pnl: float
    pnl_percent: float
    trades: int
    wins: int
    losses: int
    best_trade: float
    worst_trade: float
    volume: float


@dataclass
class MonthPerformance:
    """Performance metrics for a month."""
    year: int
    month: int
    total_pnl: float
    total_pnl_percent: float
    trading_days: int
    winning_days: int
    losing_days: int
    best_day: float
    worst_day: float
    avg_daily_pnl: float


@dataclass
class CalendarData:
    """Complete calendar data for visualization."""
    days: List[DayPerformance]
    months: List[MonthPerformance]
    yearly_summary: Dict[int, Dict[str, float]]
    heatmap_data: List[Dict[str, Any]]
    statistics: Dict[str, Any]


class CalendarViewGenerator:
    """
    Generates calendar heatmap data for backtest/trading performance.

    Creates daily performance breakdowns suitable for calendar
    heatmap visualization.
    """

    def __init__(self):
        self._daily_data: Dict[str, DayPerformance] = {}
        self._trades: List[Dict[str, Any]] = []

    def load_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Load trades for calendar generation."""
        self._trades = trades
        self._process_trades()

    def load_from_json(self, json_path: str) -> None:
        """Load trades from JSON file."""
        try:
            path = Path(json_path)
            if not path.exists():
                logger.warning(f"Trade file not found: {path}")
                return

            with open(path) as f:
                data = json.load(f)

            if isinstance(data, list):
                self._trades = data
            elif isinstance(data, dict):
                self._trades = data.get("trades", [])

            self._process_trades()

        except Exception as e:
            logger.error(f"Error loading trades: {e}")

    def _process_trades(self) -> None:
        """Process trades into daily performance data."""
        daily_trades: Dict[str, List[Dict]] = {}

        for trade in self._trades:
            # Get trade date
            timestamp = trade.get("timestamp", trade.get("close_time", ""))
            if not timestamp:
                continue

            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
            elif isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e12 else timestamp)
            else:
                continue

            date_str = dt.strftime("%Y-%m-%d")

            if date_str not in daily_trades:
                daily_trades[date_str] = []

            daily_trades[date_str].append(trade)

        # Calculate daily metrics
        for date_str, trades in daily_trades.items():
            pnls = [t.get("pnl", 0) for t in trades]
            pnl_pcts = [t.get("pnl_percent", 0) for t in trades]
            volumes = [t.get("quantity", 0) * t.get("entry_price", 0) for t in trades]

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            self._daily_data[date_str] = DayPerformance(
                date=date_str,
                pnl=sum(pnls),
                pnl_percent=sum(pnl_pcts),
                trades=len(trades),
                wins=len(wins),
                losses=len(losses),
                best_trade=max(pnls) if pnls else 0,
                worst_trade=min(pnls) if pnls else 0,
                volume=sum(volumes),
            )

    def generate_calendar(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CalendarData:
        """
        Generate calendar view data.

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            CalendarData for visualization
        """
        # Filter by date range
        filtered_days = []
        for date_str, perf in sorted(self._daily_data.items()):
            day_date = datetime.strptime(date_str, "%Y-%m-%d")

            if start_date and day_date < start_date:
                continue
            if end_date and day_date > end_date:
                continue

            filtered_days.append(perf)

        # Calculate monthly summaries
        months = self._calculate_monthly_summaries(filtered_days)

        # Calculate yearly summaries
        yearly = self._calculate_yearly_summaries(months)

        # Generate heatmap data
        heatmap = self._generate_heatmap_data(filtered_days)

        # Calculate statistics
        statistics = self._calculate_statistics(filtered_days)

        return CalendarData(
            days=filtered_days,
            months=months,
            yearly_summary=yearly,
            heatmap_data=heatmap,
            statistics=statistics,
        )

    def _calculate_monthly_summaries(
        self, days: List[DayPerformance]
    ) -> List[MonthPerformance]:
        """Calculate monthly performance summaries."""
        monthly_data: Dict[str, List[DayPerformance]] = {}

        for day in days:
            month_key = day.date[:7]  # YYYY-MM
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(day)

        months = []
        for month_key, month_days in sorted(monthly_data.items()):
            year, month = int(month_key[:4]), int(month_key[5:7])
            pnls = [d.pnl for d in month_days]
            pnl_pcts = [d.pnl_percent for d in month_days]

            winning_days = len([d for d in month_days if d.pnl > 0])
            losing_days = len([d for d in month_days if d.pnl < 0])

            months.append(MonthPerformance(
                year=year,
                month=month,
                total_pnl=sum(pnls),
                total_pnl_percent=sum(pnl_pcts),
                trading_days=len(month_days),
                winning_days=winning_days,
                losing_days=losing_days,
                best_day=max(pnls) if pnls else 0,
                worst_day=min(pnls) if pnls else 0,
                avg_daily_pnl=np.mean(pnls) if pnls else 0,
            ))

        return months

    def _calculate_yearly_summaries(
        self, months: List[MonthPerformance]
    ) -> Dict[int, Dict[str, float]]:
        """Calculate yearly performance summaries."""
        yearly: Dict[int, Dict[str, float]] = {}

        for month in months:
            if month.year not in yearly:
                yearly[month.year] = {
                    "total_pnl": 0,
                    "total_pnl_percent": 0,
                    "trading_days": 0,
                    "winning_days": 0,
                    "losing_days": 0,
                    "best_day": float("-inf"),
                    "worst_day": float("inf"),
                }

            yearly[month.year]["total_pnl"] += month.total_pnl
            yearly[month.year]["total_pnl_percent"] += month.total_pnl_percent
            yearly[month.year]["trading_days"] += month.trading_days
            yearly[month.year]["winning_days"] += month.winning_days
            yearly[month.year]["losing_days"] += month.losing_days
            yearly[month.year]["best_day"] = max(yearly[month.year]["best_day"], month.best_day)
            yearly[month.year]["worst_day"] = min(yearly[month.year]["worst_day"], month.worst_day)

        # Fix infinity values
        for year in yearly:
            if yearly[year]["best_day"] == float("-inf"):
                yearly[year]["best_day"] = 0
            if yearly[year]["worst_day"] == float("inf"):
                yearly[year]["worst_day"] = 0

        return yearly

    def _generate_heatmap_data(self, days: List[DayPerformance]) -> List[Dict[str, Any]]:
        """Generate heatmap data for calendar visualization."""
        heatmap = []

        # Determine color scale bounds
        pnls = [d.pnl_percent for d in days]
        if pnls:
            max_pnl = max(abs(min(pnls)), abs(max(pnls)), 1)
        else:
            max_pnl = 1

        for day in days:
            # Calculate color intensity (-1 to 1)
            intensity = day.pnl_percent / max_pnl if max_pnl > 0 else 0
            intensity = max(-1, min(1, intensity))

            # Parse date
            dt = datetime.strptime(day.date, "%Y-%m-%d")

            heatmap.append({
                "date": day.date,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "weekday": dt.weekday(),  # 0 = Monday
                "week": dt.isocalendar()[1],
                "pnl": round(day.pnl, 2),
                "pnl_percent": round(day.pnl_percent, 2),
                "trades": day.trades,
                "intensity": round(intensity, 3),
                "color": self._get_color(intensity),
            })

        return heatmap

    def _get_color(self, intensity: float) -> str:
        """Get color for heatmap cell based on intensity."""
        if intensity >= 0.8:
            return "#1B5E20"  # Dark green
        elif intensity >= 0.5:
            return "#2E7D32"
        elif intensity >= 0.2:
            return "#43A047"
        elif intensity >= 0.05:
            return "#66BB6A"
        elif intensity >= -0.05:
            return "#9E9E9E"  # Gray (neutral)
        elif intensity >= -0.2:
            return "#EF5350"
        elif intensity >= -0.5:
            return "#E53935"
        elif intensity >= -0.8:
            return "#C62828"
        else:
            return "#B71C1C"  # Dark red

    def _calculate_statistics(self, days: List[DayPerformance]) -> Dict[str, Any]:
        """Calculate overall statistics."""
        if not days:
            return {
                "total_days": 0,
                "winning_days": 0,
                "losing_days": 0,
                "neutral_days": 0,
                "win_rate": 0,
            }

        pnls = [d.pnl for d in days]
        pnl_pcts = [d.pnl_percent for d in days]

        winning_days = len([d for d in days if d.pnl > 0])
        losing_days = len([d for d in days if d.pnl < 0])
        neutral_days = len([d for d in days if d.pnl == 0])

        # Best/worst streaks
        best_streak, worst_streak = self._calculate_streaks(days)

        # Best day of week
        weekday_pnl: Dict[int, List[float]] = {i: [] for i in range(7)}
        for day in days:
            dt = datetime.strptime(day.date, "%Y-%m-%d")
            weekday_pnl[dt.weekday()].append(day.pnl)

        weekday_avg = {
            k: np.mean(v) if v else 0
            for k, v in weekday_pnl.items()
        }
        best_weekday = max(weekday_avg, key=weekday_avg.get)
        worst_weekday = min(weekday_avg, key=weekday_avg.get)

        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "total_days": len(days),
            "winning_days": winning_days,
            "losing_days": losing_days,
            "neutral_days": neutral_days,
            "win_rate": round((winning_days / len(days)) * 100, 1) if days else 0,
            "total_pnl": round(sum(pnls), 2),
            "total_pnl_percent": round(sum(pnl_pcts), 2),
            "avg_daily_pnl": round(np.mean(pnls), 2) if pnls else 0,
            "avg_daily_pnl_percent": round(np.mean(pnl_pcts), 2) if pnl_pcts else 0,
            "best_day": {
                "date": max(days, key=lambda d: d.pnl).date if days else None,
                "pnl": round(max(pnls), 2) if pnls else 0,
            },
            "worst_day": {
                "date": min(days, key=lambda d: d.pnl).date if days else None,
                "pnl": round(min(pnls), 2) if pnls else 0,
            },
            "best_streak": best_streak,
            "worst_streak": worst_streak,
            "best_weekday": weekday_names[best_weekday],
            "worst_weekday": weekday_names[worst_weekday],
            "weekday_performance": {
                weekday_names[k]: round(v, 2)
                for k, v in weekday_avg.items()
            },
            "volatility": round(np.std(pnl_pcts), 2) if len(pnl_pcts) > 1 else 0,
        }

    def _calculate_streaks(self, days: List[DayPerformance]) -> tuple:
        """Calculate best winning and worst losing streaks."""
        best_streak = 0
        worst_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for day in days:
            if day.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                best_streak = max(best_streak, current_win_streak)
            elif day.pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                worst_streak = max(worst_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0

        return best_streak, worst_streak

    def get_month_grid(self, year: int, month: int) -> List[List[Optional[Dict]]]:
        """
        Get calendar grid for a specific month.

        Returns 6 weeks x 7 days grid with day data.
        """
        import calendar

        cal = calendar.Calendar(firstweekday=0)  # Monday first
        grid = []

        for week in cal.monthdatescalendar(year, month):
            week_data = []
            for day in week:
                date_str = day.strftime("%Y-%m-%d")

                if day.month != month:
                    # Day from adjacent month
                    week_data.append(None)
                elif date_str in self._daily_data:
                    perf = self._daily_data[date_str]
                    week_data.append({
                        "day": day.day,
                        "date": date_str,
                        "pnl": round(perf.pnl, 2),
                        "pnl_percent": round(perf.pnl_percent, 2),
                        "trades": perf.trades,
                        "color": self._get_color(perf.pnl_percent / 5) if perf.pnl_percent != 0 else "#9E9E9E",
                    })
                else:
                    # No trading on this day
                    week_data.append({
                        "day": day.day,
                        "date": date_str,
                        "pnl": None,
                        "pnl_percent": None,
                        "trades": 0,
                        "color": "#424242",  # Dark gray for no trading
                    })

            grid.append(week_data)

        return grid

    def to_api_response(self, calendar_data: CalendarData) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "statistics": calendar_data.statistics,
            "heatmap": calendar_data.heatmap_data,
            "months": [
                {
                    "year": m.year,
                    "month": m.month,
                    "total_pnl": round(m.total_pnl, 2),
                    "total_pnl_percent": round(m.total_pnl_percent, 2),
                    "trading_days": m.trading_days,
                    "winning_days": m.winning_days,
                    "losing_days": m.losing_days,
                    "win_rate": round(m.winning_days / m.trading_days * 100, 1) if m.trading_days > 0 else 0,
                    "avg_daily_pnl": round(m.avg_daily_pnl, 2),
                }
                for m in calendar_data.months
            ],
            "yearly": {
                str(year): {k: round(v, 2) if isinstance(v, float) else v for k, v in data.items()}
                for year, data in calendar_data.yearly_summary.items()
            },
        }
