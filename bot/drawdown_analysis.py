"""
Drawdown Analysis Module.

Provides detailed drawdown visualization and analysis
including recovery times, underwater periods, and drawdown distribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DrawdownPeriod:
    """Represents a single drawdown period."""
    start_date: datetime
    end_date: Optional[datetime]
    recovery_date: Optional[datetime]
    peak_value: float
    trough_value: float
    drawdown_percent: float
    duration_days: float
    recovery_days: Optional[float]
    is_recovered: bool
    peak_index: int
    trough_index: int


@dataclass
class DrawdownAnalysis:
    """Complete drawdown analysis results."""
    max_drawdown: float
    max_drawdown_date: Optional[datetime]
    max_drawdown_duration: float
    avg_drawdown: float
    avg_recovery_time: float
    current_drawdown: float
    current_drawdown_start: Optional[datetime]
    drawdown_periods: List[DrawdownPeriod]
    underwater_curve: List[Dict[str, Any]]
    drawdown_distribution: Dict[str, int]
    recovery_factor: float
    ulcer_index: float
    pain_index: float
    calmar_ratio: float
    time_in_drawdown_percent: float


class DrawdownAnalyzer:
    """
    Analyzes portfolio drawdowns with comprehensive metrics.

    Calculates max drawdown, recovery times, underwater periods,
    and provides data for visualization.
    """

    def __init__(self):
        self._equity_curve: List[float] = []
        self._dates: List[datetime] = []

    def load_equity_from_json(self, json_path: str) -> None:
        """Load equity curve from JSON file."""
        try:
            path = Path(json_path)
            if not path.exists():
                logger.warning(f"Equity file not found: {path}")
                return

            with open(path) as f:
                data = json.load(f)

            if isinstance(data, list):
                # List of equity values
                self._equity_curve = [float(v) for v in data]
                self._dates = [
                    datetime.now() - timedelta(hours=len(data) - i)
                    for i in range(len(data))
                ]
            elif isinstance(data, dict):
                # Dict with timestamps
                curve = data.get("curve", data.get("equity", []))
                if isinstance(curve, list) and curve:
                    if isinstance(curve[0], dict):
                        self._equity_curve = [float(p.get("value", p.get("equity", 0))) for p in curve]
                        self._dates = [
                            datetime.fromisoformat(p.get("timestamp", p.get("date", "")))
                            if p.get("timestamp") or p.get("date")
                            else datetime.now() - timedelta(hours=len(curve) - i)
                            for i, p in enumerate(curve)
                        ]
                    else:
                        self._equity_curve = [float(v) for v in curve]
                        self._dates = [
                            datetime.now() - timedelta(hours=len(curve) - i)
                            for i in range(len(curve))
                        ]

        except Exception as e:
            logger.error(f"Error loading equity curve: {e}")

    def load_equity_curve(
        self,
        equity_values: List[float],
        dates: Optional[List[datetime]] = None,
    ) -> None:
        """Load equity curve from values."""
        self._equity_curve = equity_values
        if dates:
            self._dates = dates
        else:
            # Generate dates (assume hourly)
            self._dates = [
                datetime.now() - timedelta(hours=len(equity_values) - i)
                for i in range(len(equity_values))
            ]

    def analyze(self, annual_return: Optional[float] = None) -> DrawdownAnalysis:
        """
        Perform comprehensive drawdown analysis.

        Args:
            annual_return: Annualized return for Calmar ratio calculation

        Returns:
            DrawdownAnalysis with all metrics and periods
        """
        if not self._equity_curve or len(self._equity_curve) < 2:
            return DrawdownAnalysis(
                max_drawdown=0,
                max_drawdown_date=None,
                max_drawdown_duration=0,
                avg_drawdown=0,
                avg_recovery_time=0,
                current_drawdown=0,
                current_drawdown_start=None,
                drawdown_periods=[],
                underwater_curve=[],
                drawdown_distribution={},
                recovery_factor=0,
                ulcer_index=0,
                pain_index=0,
                calmar_ratio=0,
                time_in_drawdown_percent=0,
            )

        equity = np.array(self._equity_curve)
        dates = self._dates

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown series
        drawdown = (running_max - equity) / running_max
        drawdown = np.nan_to_num(drawdown, nan=0.0)

        # Find drawdown periods
        periods = self._find_drawdown_periods(equity, running_max, drawdown, dates)

        # Calculate max drawdown
        max_dd = float(np.max(drawdown)) * 100
        max_dd_idx = int(np.argmax(drawdown))
        max_dd_date = dates[max_dd_idx] if max_dd_idx < len(dates) else None

        # Calculate max drawdown duration
        max_dd_duration = 0.0
        if periods:
            max_dd_duration = max(p.duration_days for p in periods)

        # Average drawdown (excluding zero periods)
        non_zero_dd = drawdown[drawdown > 0.001]
        avg_dd = float(np.mean(non_zero_dd)) * 100 if len(non_zero_dd) > 0 else 0

        # Average recovery time
        recovered_periods = [p for p in periods if p.is_recovered and p.recovery_days is not None]
        avg_recovery = (
            np.mean([p.recovery_days for p in recovered_periods])
            if recovered_periods
            else 0
        )

        # Current drawdown
        current_dd = float(drawdown[-1]) * 100
        current_dd_start = None
        if current_dd > 0.1:  # More than 0.1% drawdown
            # Find when current drawdown started
            for i in range(len(drawdown) - 1, -1, -1):
                if drawdown[i] < 0.001:
                    if i + 1 < len(dates):
                        current_dd_start = dates[i + 1]
                    break

        # Underwater curve (drawdown over time)
        underwater_curve = [
            {
                "date": dates[i].isoformat() if i < len(dates) else "",
                "drawdown": round(float(drawdown[i]) * 100, 2),
                "equity": round(float(equity[i]), 2),
                "peak": round(float(running_max[i]), 2),
            }
            for i in range(len(drawdown))
        ]

        # Drawdown distribution
        distribution = self._calculate_drawdown_distribution(drawdown)

        # Recovery factor
        total_return = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else 0
        recovery_factor = total_return / max_dd if max_dd > 0 else 0

        # Ulcer Index (RMS of drawdown)
        ulcer_index = float(np.sqrt(np.mean(drawdown ** 2))) * 100

        # Pain Index (mean of drawdown)
        pain_index = float(np.mean(drawdown)) * 100

        # Calmar Ratio
        calmar_ratio = 0.0
        if annual_return is not None and max_dd > 0:
            calmar_ratio = annual_return / max_dd
        elif max_dd > 0 and len(equity) > 252:
            # Estimate annual return
            ann_return = (equity[-1] / equity[0]) ** (252 / len(equity)) - 1
            calmar_ratio = (ann_return * 100) / max_dd

        # Time in drawdown
        time_in_dd = (np.sum(drawdown > 0.001) / len(drawdown)) * 100

        return DrawdownAnalysis(
            max_drawdown=round(max_dd, 2),
            max_drawdown_date=max_dd_date,
            max_drawdown_duration=round(max_dd_duration, 1),
            avg_drawdown=round(avg_dd, 2),
            avg_recovery_time=round(avg_recovery, 1),
            current_drawdown=round(current_dd, 2),
            current_drawdown_start=current_dd_start,
            drawdown_periods=periods,
            underwater_curve=underwater_curve,
            drawdown_distribution=distribution,
            recovery_factor=round(recovery_factor, 2),
            ulcer_index=round(ulcer_index, 2),
            pain_index=round(pain_index, 2),
            calmar_ratio=round(calmar_ratio, 2),
            time_in_drawdown_percent=round(time_in_dd, 1),
        )

    def _find_drawdown_periods(
        self,
        equity: np.ndarray,
        running_max: np.ndarray,
        drawdown: np.ndarray,
        dates: List[datetime],
    ) -> List[DrawdownPeriod]:
        """Identify individual drawdown periods."""
        periods = []
        in_drawdown = False
        peak_idx = 0
        trough_idx = 0
        peak_value = 0.0
        trough_value = float("inf")

        threshold = 0.001  # 0.1% threshold

        for i in range(len(drawdown)):
            if not in_drawdown and drawdown[i] > threshold:
                # Start of drawdown
                in_drawdown = True
                peak_idx = i - 1 if i > 0 else 0
                peak_value = running_max[i]
                trough_idx = i
                trough_value = equity[i]

            elif in_drawdown:
                if equity[i] < trough_value:
                    # New trough
                    trough_idx = i
                    trough_value = equity[i]

                if drawdown[i] < threshold:
                    # Recovery
                    dd_percent = (peak_value - trough_value) / peak_value * 100 if peak_value > 0 else 0

                    period = DrawdownPeriod(
                        start_date=dates[peak_idx] if peak_idx < len(dates) else datetime.now(),
                        end_date=dates[trough_idx] if trough_idx < len(dates) else datetime.now(),
                        recovery_date=dates[i] if i < len(dates) else datetime.now(),
                        peak_value=float(peak_value),
                        trough_value=float(trough_value),
                        drawdown_percent=round(dd_percent, 2),
                        duration_days=(dates[trough_idx] - dates[peak_idx]).days if trough_idx < len(dates) and peak_idx < len(dates) else 0,
                        recovery_days=(dates[i] - dates[trough_idx]).days if i < len(dates) and trough_idx < len(dates) else 0,
                        is_recovered=True,
                        peak_index=peak_idx,
                        trough_index=trough_idx,
                    )
                    periods.append(period)
                    in_drawdown = False

        # Handle ongoing drawdown
        if in_drawdown:
            dd_percent = (peak_value - trough_value) / peak_value * 100 if peak_value > 0 else 0

            period = DrawdownPeriod(
                start_date=dates[peak_idx] if peak_idx < len(dates) else datetime.now(),
                end_date=dates[trough_idx] if trough_idx < len(dates) else datetime.now(),
                recovery_date=None,
                peak_value=float(peak_value),
                trough_value=float(trough_value),
                drawdown_percent=round(dd_percent, 2),
                duration_days=(dates[-1] - dates[peak_idx]).days if peak_idx < len(dates) else 0,
                recovery_days=None,
                is_recovered=False,
                peak_index=peak_idx,
                trough_index=trough_idx,
            )
            periods.append(period)

        return periods

    def _calculate_drawdown_distribution(self, drawdown: np.ndarray) -> Dict[str, int]:
        """Calculate distribution of drawdown magnitudes."""
        distribution = {
            "0-1%": 0,
            "1-5%": 0,
            "5-10%": 0,
            "10-20%": 0,
            "20-30%": 0,
            "30%+": 0,
        }

        dd_percent = drawdown * 100

        for dd in dd_percent:
            if dd < 1:
                distribution["0-1%"] += 1
            elif dd < 5:
                distribution["1-5%"] += 1
            elif dd < 10:
                distribution["5-10%"] += 1
            elif dd < 20:
                distribution["10-20%"] += 1
            elif dd < 30:
                distribution["20-30%"] += 1
            else:
                distribution["30%+"] += 1

        return distribution

    def get_top_drawdowns(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N largest drawdown periods."""
        analysis = self.analyze()
        sorted_periods = sorted(
            analysis.drawdown_periods,
            key=lambda p: p.drawdown_percent,
            reverse=True,
        )[:n]

        return [
            {
                "rank": i + 1,
                "start_date": p.start_date.strftime("%Y-%m-%d") if p.start_date else "N/A",
                "end_date": p.end_date.strftime("%Y-%m-%d") if p.end_date else "N/A",
                "recovery_date": p.recovery_date.strftime("%Y-%m-%d") if p.recovery_date else "Ongoing",
                "drawdown_percent": p.drawdown_percent,
                "duration_days": p.duration_days,
                "recovery_days": p.recovery_days if p.recovery_days else "N/A",
                "is_recovered": p.is_recovered,
            }
            for i, p in enumerate(sorted_periods)
        ]

    def to_api_response(self, analysis: DrawdownAnalysis) -> Dict[str, Any]:
        """Convert analysis to API response format."""
        return {
            "summary": {
                "max_drawdown": analysis.max_drawdown,
                "max_drawdown_date": analysis.max_drawdown_date.isoformat() if analysis.max_drawdown_date else None,
                "max_drawdown_duration_days": analysis.max_drawdown_duration,
                "avg_drawdown": analysis.avg_drawdown,
                "avg_recovery_days": analysis.avg_recovery_time,
                "current_drawdown": analysis.current_drawdown,
                "current_drawdown_start": analysis.current_drawdown_start.isoformat() if analysis.current_drawdown_start else None,
            },
            "risk_metrics": {
                "recovery_factor": analysis.recovery_factor,
                "ulcer_index": analysis.ulcer_index,
                "pain_index": analysis.pain_index,
                "calmar_ratio": analysis.calmar_ratio,
                "time_in_drawdown_percent": analysis.time_in_drawdown_percent,
            },
            "distribution": analysis.drawdown_distribution,
            "underwater_curve": analysis.underwater_curve[-100:] if len(analysis.underwater_curve) > 100 else analysis.underwater_curve,
            "top_drawdowns": [
                {
                    "rank": i + 1,
                    "start_date": p.start_date.strftime("%Y-%m-%d") if p.start_date else None,
                    "drawdown_percent": p.drawdown_percent,
                    "duration_days": p.duration_days,
                    "recovery_days": p.recovery_days,
                    "is_recovered": p.is_recovered,
                }
                for i, p in enumerate(
                    sorted(analysis.drawdown_periods, key=lambda x: x.drawdown_percent, reverse=True)[:5]
                )
            ],
            "total_drawdown_periods": len(analysis.drawdown_periods),
            "recovered_periods": len([p for p in analysis.drawdown_periods if p.is_recovered]),
        }
