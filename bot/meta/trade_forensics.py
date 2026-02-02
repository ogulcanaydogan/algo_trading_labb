"""
Post-Trade Forensics Module.

Provides deep analysis of completed trades to identify:
- MAE (Maximum Adverse Excursion) - worst point during trade
- MFE (Maximum Favorable Excursion) - best point during trade
- Stop quality analysis
- Entry timing quality
- Exit timing quality
- Holding period optimization

Used for continuous strategy improvement and learning.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExitQuality(Enum):
    """Quality rating for trade exits."""

    OPTIMAL = "optimal"  # Within 5% of MFE
    GOOD = "good"  # Within 15% of MFE
    ACCEPTABLE = "acceptable"  # Within 30% of MFE
    POOR = "poor"  # Beyond 30% from MFE
    STOPPED_OUT = "stopped_out"  # Hit stop loss


class EntryQuality(Enum):
    """Quality rating for trade entries."""

    OPTIMAL = "optimal"  # Entry within 5% of best entry point
    GOOD = "good"  # Entry within 15%
    ACCEPTABLE = "acceptable"  # Entry within 30%
    POOR = "poor"  # Entry beyond 30% from optimal


class StopQuality(Enum):
    """Quality rating for stop loss placement."""

    TOO_TIGHT = "too_tight"  # Stop was too close, premature exit
    OPTIMAL = "optimal"  # Stop was well-placed
    TOO_LOOSE = "too_loose"  # Stop was too far, excess risk
    NOT_USED = "not_used"  # No stop loss was used


@dataclass
class PricePoint:
    """Price and timestamp during a trade."""

    timestamp: datetime
    price: float
    unrealized_pnl: float  # P&L at this point
    unrealized_pnl_pct: float


@dataclass
class ForensicsResult:
    """Complete forensics analysis for a single trade."""

    trade_id: str
    symbol: str
    side: str  # "long" or "short"

    # Entry analysis
    entry_price: float
    entry_timestamp: datetime
    entry_quality: EntryQuality
    entry_quality_score: float  # 0-1 score

    # Exit analysis
    exit_price: float
    exit_timestamp: datetime
    exit_quality: ExitQuality
    exit_quality_score: float  # 0-1 score

    # MAE/MFE analysis
    mae_price: float  # Price at maximum adverse excursion
    mae_pnl: float  # P&L at MAE
    mae_pct: float  # Percentage at MAE
    mae_timestamp: datetime

    mfe_price: float  # Price at maximum favorable excursion
    mfe_pnl: float  # P&L at MFE
    mfe_pct: float  # Percentage at MFE
    mfe_timestamp: datetime

    # Stop analysis
    stop_price: Optional[float]
    stop_quality: StopQuality
    stop_quality_score: float  # 0-1 score
    optimal_stop_price: float  # Where stop should have been

    # Timing analysis
    time_to_mae_minutes: float
    time_to_mfe_minutes: float
    holding_period_minutes: float
    optimal_holding_minutes: float  # Time to MFE

    # Final P&L
    realized_pnl: float
    realized_pnl_pct: float

    # Efficiency metrics
    capture_ratio: float  # Realized / MFE (how much of potential captured)
    pain_ratio: float  # MAE / MFE (pain vs reward)
    edge_efficiency: float  # Overall efficiency score

    # Recommendations
    improvements: List[str] = field(default_factory=list)


@dataclass
class ForensicsConfig:
    """Configuration for forensics analysis."""

    # Quality thresholds (percentage of MFE)
    optimal_exit_threshold: float = 0.95  # 95% of MFE
    good_exit_threshold: float = 0.85
    acceptable_exit_threshold: float = 0.70

    # Entry quality thresholds (percentage from optimal)
    optimal_entry_threshold: float = 0.05  # Within 5%
    good_entry_threshold: float = 0.15
    acceptable_entry_threshold: float = 0.30

    # Stop quality thresholds
    tight_stop_threshold: float = 0.3  # MAE < 30% of stop distance = too tight
    loose_stop_threshold: float = 2.0  # MAE > 200% of stop distance = too loose

    # Minimum price points for analysis
    min_price_points: int = 5

    # Historical storage
    max_history_trades: int = 1000
    state_path: Path = field(default_factory=lambda: Path("data/trade_forensics_state.json"))


@dataclass
class AggregateStats:
    """Aggregate forensics statistics."""

    total_trades: int = 0
    avg_capture_ratio: float = 0.0
    avg_pain_ratio: float = 0.0
    avg_edge_efficiency: float = 0.0

    # Exit quality distribution
    optimal_exits: int = 0
    good_exits: int = 0
    acceptable_exits: int = 0
    poor_exits: int = 0
    stopped_out_exits: int = 0

    # Entry quality distribution
    optimal_entries: int = 0
    good_entries: int = 0
    acceptable_entries: int = 0
    poor_entries: int = 0

    # Stop quality distribution
    tight_stops: int = 0
    optimal_stops: int = 0
    loose_stops: int = 0
    no_stops: int = 0

    # Timing stats
    avg_mae_minutes: float = 0.0
    avg_mfe_minutes: float = 0.0
    avg_holding_minutes: float = 0.0
    avg_optimal_holding_minutes: float = 0.0


class TradeForensics:
    """
    Post-trade forensics analyzer.

    Analyzes completed trades to measure execution quality
    and provide improvement recommendations.

    Thread-safe singleton pattern.
    """

    _instance: Optional["TradeForensics"] = None
    _lock = RLock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ForensicsConfig] = None):
        if self._initialized:
            return

        self._config = config or ForensicsConfig()
        self._lock = RLock()

        # History storage
        self._trade_history: List[ForensicsResult] = []
        self._aggregate_stats = AggregateStats()

        self._load_state()
        self._initialized = True

        logger.info("TradeForensics initialized")

    def _load_state(self) -> None:
        """Load persisted state."""
        if self._config.state_path.exists():
            try:
                with open(self._config.state_path, "r") as f:
                    data = json.load(f)

                # Load aggregate stats
                stats_data = data.get("aggregate_stats", {})
                self._aggregate_stats = AggregateStats(**stats_data)

                logger.info(
                    f"Loaded forensics state: {self._aggregate_stats.total_trades} trades analyzed"
                )

            except Exception as e:
                logger.warning(f"Failed to load forensics state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self._config.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config.state_path, "w") as f:
                json.dump(
                    {
                        "aggregate_stats": {
                            "total_trades": self._aggregate_stats.total_trades,
                            "avg_capture_ratio": self._aggregate_stats.avg_capture_ratio,
                            "avg_pain_ratio": self._aggregate_stats.avg_pain_ratio,
                            "avg_edge_efficiency": self._aggregate_stats.avg_edge_efficiency,
                            "optimal_exits": self._aggregate_stats.optimal_exits,
                            "good_exits": self._aggregate_stats.good_exits,
                            "acceptable_exits": self._aggregate_stats.acceptable_exits,
                            "poor_exits": self._aggregate_stats.poor_exits,
                            "stopped_out_exits": self._aggregate_stats.stopped_out_exits,
                            "optimal_entries": self._aggregate_stats.optimal_entries,
                            "good_entries": self._aggregate_stats.good_entries,
                            "acceptable_entries": self._aggregate_stats.acceptable_entries,
                            "poor_entries": self._aggregate_stats.poor_entries,
                            "tight_stops": self._aggregate_stats.tight_stops,
                            "optimal_stops": self._aggregate_stats.optimal_stops,
                            "loose_stops": self._aggregate_stats.loose_stops,
                            "no_stops": self._aggregate_stats.no_stops,
                            "avg_mae_minutes": self._aggregate_stats.avg_mae_minutes,
                            "avg_mfe_minutes": self._aggregate_stats.avg_mfe_minutes,
                            "avg_holding_minutes": self._aggregate_stats.avg_holding_minutes,
                            "avg_optimal_holding_minutes": self._aggregate_stats.avg_optimal_holding_minutes,
                        },
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save forensics state: {e}")

    def analyze_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        entry_timestamp: datetime,
        exit_price: float,
        exit_timestamp: datetime,
        stop_price: Optional[float],
        price_history: List[Tuple[datetime, float]],
        was_stopped_out: bool = False,
    ) -> ForensicsResult:
        """
        Perform forensics analysis on a completed trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Price at entry
            entry_timestamp: Time of entry
            exit_price: Price at exit
            exit_timestamp: Time of exit
            stop_price: Stop loss price (if any)
            price_history: List of (timestamp, price) during trade
            was_stopped_out: Whether trade hit stop loss

        Returns:
            ForensicsResult with complete analysis
        """
        with self._lock:
            # Convert price history to PricePoints
            price_points = self._build_price_points(
                side, entry_price, price_history
            )

            if len(price_points) < self._config.min_price_points:
                logger.warning(
                    f"Insufficient price points for trade {trade_id}: {len(price_points)}"
                )

            # Calculate MAE and MFE
            mae_point = self._calculate_mae(side, price_points)
            mfe_point = self._calculate_mfe(side, price_points)

            # Calculate P&L
            if side == "long":
                realized_pnl = exit_price - entry_price
                realized_pnl_pct = realized_pnl / entry_price
            else:  # short
                realized_pnl = entry_price - exit_price
                realized_pnl_pct = realized_pnl / entry_price

            # Calculate holding period
            holding_period = (exit_timestamp - entry_timestamp).total_seconds() / 60

            # Calculate time to MAE/MFE
            time_to_mae = (mae_point.timestamp - entry_timestamp).total_seconds() / 60
            time_to_mfe = (mfe_point.timestamp - entry_timestamp).total_seconds() / 60

            # Analyze entry quality
            entry_quality, entry_score = self._analyze_entry_quality(
                side, entry_price, price_points
            )

            # Analyze exit quality
            exit_quality, exit_score = self._analyze_exit_quality(
                side, exit_price, mfe_point, was_stopped_out
            )

            # Analyze stop quality
            stop_quality, stop_score, optimal_stop = self._analyze_stop_quality(
                side, entry_price, stop_price, mae_point, mfe_point
            )

            # Calculate efficiency metrics
            if abs(mfe_point.unrealized_pnl) > 0:
                capture_ratio = realized_pnl / mfe_point.unrealized_pnl
            else:
                capture_ratio = 0.0 if realized_pnl <= 0 else 1.0

            if abs(mfe_point.unrealized_pnl) > 0:
                pain_ratio = abs(mae_point.unrealized_pnl) / abs(mfe_point.unrealized_pnl)
            else:
                pain_ratio = 1.0

            # Edge efficiency combines entry, exit, and stop quality
            edge_efficiency = (
                entry_score * 0.3 + exit_score * 0.5 + stop_score * 0.2
            )

            # Generate improvement recommendations
            improvements = self._generate_recommendations(
                entry_quality,
                exit_quality,
                stop_quality,
                capture_ratio,
                pain_ratio,
                time_to_mae,
                time_to_mfe,
                holding_period,
            )

            result = ForensicsResult(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                entry_timestamp=entry_timestamp,
                entry_quality=entry_quality,
                entry_quality_score=entry_score,
                exit_price=exit_price,
                exit_timestamp=exit_timestamp,
                exit_quality=exit_quality,
                exit_quality_score=exit_score,
                mae_price=mae_point.price,
                mae_pnl=mae_point.unrealized_pnl,
                mae_pct=mae_point.unrealized_pnl_pct,
                mae_timestamp=mae_point.timestamp,
                mfe_price=mfe_point.price,
                mfe_pnl=mfe_point.unrealized_pnl,
                mfe_pct=mfe_point.unrealized_pnl_pct,
                mfe_timestamp=mfe_point.timestamp,
                stop_price=stop_price,
                stop_quality=stop_quality,
                stop_quality_score=stop_score,
                optimal_stop_price=optimal_stop,
                time_to_mae_minutes=time_to_mae,
                time_to_mfe_minutes=time_to_mfe,
                holding_period_minutes=holding_period,
                optimal_holding_minutes=time_to_mfe,
                realized_pnl=realized_pnl,
                realized_pnl_pct=realized_pnl_pct,
                capture_ratio=capture_ratio,
                pain_ratio=pain_ratio,
                edge_efficiency=edge_efficiency,
                improvements=improvements,
            )

            # Update history and stats
            self._update_history(result)

            logger.info(
                f"Forensics for {trade_id}: capture={capture_ratio:.1%}, "
                f"pain={pain_ratio:.1%}, efficiency={edge_efficiency:.1%}"
            )

            return result

    def _build_price_points(
        self,
        side: str,
        entry_price: float,
        price_history: List[Tuple[datetime, float]],
    ) -> List[PricePoint]:
        """Build PricePoint list with P&L calculations."""
        points = []

        for ts, price in price_history:
            if side == "long":
                unrealized_pnl = price - entry_price
            else:  # short
                unrealized_pnl = entry_price - price

            unrealized_pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0.0

            points.append(
                PricePoint(
                    timestamp=ts,
                    price=price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                )
            )

        return sorted(points, key=lambda p: p.timestamp)

    def _calculate_mae(self, side: str, price_points: List[PricePoint]) -> PricePoint:
        """Find Maximum Adverse Excursion point."""
        if not price_points:
            now = datetime.now()
            return PricePoint(now, 0.0, 0.0, 0.0)

        # MAE is the point with minimum (worst) P&L
        return min(price_points, key=lambda p: p.unrealized_pnl)

    def _calculate_mfe(self, side: str, price_points: List[PricePoint]) -> PricePoint:
        """Find Maximum Favorable Excursion point."""
        if not price_points:
            now = datetime.now()
            return PricePoint(now, 0.0, 0.0, 0.0)

        # MFE is the point with maximum (best) P&L
        return max(price_points, key=lambda p: p.unrealized_pnl)

    def _analyze_entry_quality(
        self,
        side: str,
        entry_price: float,
        price_points: List[PricePoint],
    ) -> Tuple[EntryQuality, float]:
        """Analyze quality of trade entry."""
        if not price_points:
            return EntryQuality.ACCEPTABLE, 0.5

        # Find the best entry point (lowest for long, highest for short)
        if side == "long":
            optimal_entry = min(p.price for p in price_points)
        else:
            optimal_entry = max(p.price for p in price_points)

        # Calculate deviation from optimal
        if optimal_entry > 0:
            deviation = abs(entry_price - optimal_entry) / optimal_entry
        else:
            deviation = 0.0

        # Determine quality
        if deviation <= self._config.optimal_entry_threshold:
            quality = EntryQuality.OPTIMAL
            score = 1.0 - (deviation / self._config.optimal_entry_threshold) * 0.1
        elif deviation <= self._config.good_entry_threshold:
            quality = EntryQuality.GOOD
            score = 0.8 - (
                (deviation - self._config.optimal_entry_threshold)
                / (self._config.good_entry_threshold - self._config.optimal_entry_threshold)
            ) * 0.2
        elif deviation <= self._config.acceptable_entry_threshold:
            quality = EntryQuality.ACCEPTABLE
            score = 0.5 - (
                (deviation - self._config.good_entry_threshold)
                / (self._config.acceptable_entry_threshold - self._config.good_entry_threshold)
            ) * 0.2
        else:
            quality = EntryQuality.POOR
            score = max(0.0, 0.3 - (deviation - self._config.acceptable_entry_threshold))

        return quality, max(0.0, min(1.0, score))

    def _analyze_exit_quality(
        self,
        side: str,
        exit_price: float,
        mfe_point: PricePoint,
        was_stopped_out: bool,
    ) -> Tuple[ExitQuality, float]:
        """Analyze quality of trade exit."""
        if was_stopped_out:
            return ExitQuality.STOPPED_OUT, 0.3

        if abs(mfe_point.unrealized_pnl) < 0.0001:
            return ExitQuality.ACCEPTABLE, 0.5

        # Calculate how close exit was to MFE
        if side == "long":
            exit_pnl = exit_price - (mfe_point.price - mfe_point.unrealized_pnl)
        else:
            entry_for_mfe = mfe_point.price + mfe_point.unrealized_pnl
            exit_pnl = entry_for_mfe - exit_price

        if mfe_point.unrealized_pnl > 0:
            capture_ratio = exit_pnl / mfe_point.unrealized_pnl
        else:
            capture_ratio = 0.0

        # Clamp ratio to [0, 1] range
        capture_ratio = max(0.0, min(1.0, capture_ratio))

        # Determine quality
        if capture_ratio >= self._config.optimal_exit_threshold:
            quality = ExitQuality.OPTIMAL
            score = 0.95 + (capture_ratio - self._config.optimal_exit_threshold) * 0.1
        elif capture_ratio >= self._config.good_exit_threshold:
            quality = ExitQuality.GOOD
            score = 0.75 + (
                (capture_ratio - self._config.good_exit_threshold)
                / (self._config.optimal_exit_threshold - self._config.good_exit_threshold)
            ) * 0.2
        elif capture_ratio >= self._config.acceptable_exit_threshold:
            quality = ExitQuality.ACCEPTABLE
            score = 0.5 + (
                (capture_ratio - self._config.acceptable_exit_threshold)
                / (self._config.good_exit_threshold - self._config.acceptable_exit_threshold)
            ) * 0.25
        else:
            quality = ExitQuality.POOR
            score = capture_ratio / self._config.acceptable_exit_threshold * 0.5

        return quality, max(0.0, min(1.0, score))

    def _analyze_stop_quality(
        self,
        side: str,
        entry_price: float,
        stop_price: Optional[float],
        mae_point: PricePoint,
        mfe_point: PricePoint,
    ) -> Tuple[StopQuality, float, float]:
        """
        Analyze quality of stop loss placement.

        Returns (quality, score, optimal_stop_price).
        """
        if stop_price is None or stop_price <= 0:
            # Calculate where stop should have been
            if side == "long":
                optimal_stop = mae_point.price * 0.99  # Just below MAE
            else:
                optimal_stop = mae_point.price * 1.01  # Just above MAE

            return StopQuality.NOT_USED, 0.5, optimal_stop

        # Calculate stop distance from entry
        if side == "long":
            stop_distance = entry_price - stop_price
            mae_distance = entry_price - mae_point.price
        else:
            stop_distance = stop_price - entry_price
            mae_distance = mae_point.price - entry_price

        # Prevent division by zero
        if stop_distance <= 0:
            if side == "long":
                optimal_stop = mae_point.price * 0.99
            else:
                optimal_stop = mae_point.price * 1.01
            return StopQuality.TOO_TIGHT, 0.3, optimal_stop

        # Calculate ratio: MAE/stop_distance
        # If ratio < 0.3: stop was too tight (would have been hit prematurely)
        # If ratio > 2.0: stop was too loose (excess risk)
        # Optimal: ratio between 0.5 and 1.0
        ratio = mae_distance / stop_distance

        # Optimal stop: just beyond MAE (10% buffer)
        if side == "long":
            optimal_stop = mae_point.price * 0.99
        else:
            optimal_stop = mae_point.price * 1.01

        if ratio < self._config.tight_stop_threshold:
            quality = StopQuality.TOO_TIGHT
            score = 0.3 + ratio / self._config.tight_stop_threshold * 0.2
        elif ratio > self._config.loose_stop_threshold:
            quality = StopQuality.TOO_LOOSE
            score = max(0.2, 0.5 - (ratio - self._config.loose_stop_threshold) * 0.1)
        else:
            quality = StopQuality.OPTIMAL
            # Best score when ratio is around 0.6-0.8
            optimal_ratio = 0.7
            score = 0.9 - abs(ratio - optimal_ratio) * 0.3

        return quality, max(0.0, min(1.0, score)), optimal_stop

    def _generate_recommendations(
        self,
        entry_quality: EntryQuality,
        exit_quality: ExitQuality,
        stop_quality: StopQuality,
        capture_ratio: float,
        pain_ratio: float,
        time_to_mae: float,
        time_to_mfe: float,
        holding_period: float,
    ) -> List[str]:
        """Generate specific improvement recommendations."""
        improvements = []

        # Entry recommendations
        if entry_quality == EntryQuality.POOR:
            improvements.append(
                "Entry timing needs improvement - consider waiting for better price action"
            )
        elif entry_quality == EntryQuality.ACCEPTABLE:
            improvements.append(
                "Entry timing acceptable but could be optimized with better confirmation"
            )

        # Exit recommendations
        if exit_quality == ExitQuality.POOR:
            improvements.append(
                f"Exit captured only {capture_ratio:.0%} of potential - consider trailing stops"
            )
        elif exit_quality == ExitQuality.STOPPED_OUT:
            if stop_quality == StopQuality.TOO_TIGHT:
                improvements.append(
                    "Stopped out due to tight stop - widen stop or improve entry timing"
                )

        # Stop recommendations
        if stop_quality == StopQuality.TOO_TIGHT:
            improvements.append(
                "Stop loss was too tight relative to market volatility"
            )
        elif stop_quality == StopQuality.TOO_LOOSE:
            improvements.append(
                "Stop loss was too wide - exposed to excess risk"
            )
        elif stop_quality == StopQuality.NOT_USED:
            improvements.append(
                "Consider using stop losses for risk management"
            )

        # Pain ratio recommendations
        if pain_ratio > 0.8:
            improvements.append(
                f"High pain ratio ({pain_ratio:.0%}) - trade experienced significant drawdown relative to profit"
            )

        # Timing recommendations
        if time_to_mae < time_to_mfe and time_to_mae < holding_period * 0.2:
            improvements.append(
                "Immediate adverse move after entry - consider improving entry confirmation"
            )

        if time_to_mfe > 0 and holding_period > time_to_mfe * 2:
            improvements.append(
                f"Held {holding_period/time_to_mfe:.1f}x longer than optimal - "
                "consider time-based exit rules"
            )

        # Low capture recommendations
        if capture_ratio < 0.5 and exit_quality != ExitQuality.STOPPED_OUT:
            improvements.append(
                "Left significant profit on the table - "
                "consider partial profit-taking or trailing stops"
            )

        return improvements

    def _update_history(self, result: ForensicsResult) -> None:
        """Update history and aggregate statistics."""
        # Add to history (limit size)
        self._trade_history.append(result)
        if len(self._trade_history) > self._config.max_history_trades:
            self._trade_history = self._trade_history[-self._config.max_history_trades :]

        # Update aggregate stats incrementally
        stats = self._aggregate_stats
        n = stats.total_trades
        n_new = n + 1

        # Running averages
        stats.avg_capture_ratio = (stats.avg_capture_ratio * n + result.capture_ratio) / n_new
        stats.avg_pain_ratio = (stats.avg_pain_ratio * n + result.pain_ratio) / n_new
        stats.avg_edge_efficiency = (
            stats.avg_edge_efficiency * n + result.edge_efficiency
        ) / n_new
        stats.avg_mae_minutes = (
            stats.avg_mae_minutes * n + result.time_to_mae_minutes
        ) / n_new
        stats.avg_mfe_minutes = (
            stats.avg_mfe_minutes * n + result.time_to_mfe_minutes
        ) / n_new
        stats.avg_holding_minutes = (
            stats.avg_holding_minutes * n + result.holding_period_minutes
        ) / n_new
        stats.avg_optimal_holding_minutes = (
            stats.avg_optimal_holding_minutes * n + result.optimal_holding_minutes
        ) / n_new

        stats.total_trades = n_new

        # Update quality distributions
        if result.exit_quality == ExitQuality.OPTIMAL:
            stats.optimal_exits += 1
        elif result.exit_quality == ExitQuality.GOOD:
            stats.good_exits += 1
        elif result.exit_quality == ExitQuality.ACCEPTABLE:
            stats.acceptable_exits += 1
        elif result.exit_quality == ExitQuality.POOR:
            stats.poor_exits += 1
        elif result.exit_quality == ExitQuality.STOPPED_OUT:
            stats.stopped_out_exits += 1

        if result.entry_quality == EntryQuality.OPTIMAL:
            stats.optimal_entries += 1
        elif result.entry_quality == EntryQuality.GOOD:
            stats.good_entries += 1
        elif result.entry_quality == EntryQuality.ACCEPTABLE:
            stats.acceptable_entries += 1
        elif result.entry_quality == EntryQuality.POOR:
            stats.poor_entries += 1

        if result.stop_quality == StopQuality.TOO_TIGHT:
            stats.tight_stops += 1
        elif result.stop_quality == StopQuality.OPTIMAL:
            stats.optimal_stops += 1
        elif result.stop_quality == StopQuality.TOO_LOOSE:
            stats.loose_stops += 1
        elif result.stop_quality == StopQuality.NOT_USED:
            stats.no_stops += 1

        self._save_state()

    def get_aggregate_stats(self) -> AggregateStats:
        """Get aggregate statistics."""
        with self._lock:
            return self._aggregate_stats

    def get_recent_analysis(self, n: int = 10) -> List[ForensicsResult]:
        """Get N most recent forensics results."""
        with self._lock:
            return list(self._trade_history[-n:])

    def get_improvement_priorities(self) -> List[Tuple[str, float]]:
        """
        Get prioritized list of improvements based on aggregate data.

        Returns list of (improvement_area, impact_score) sorted by impact.
        """
        with self._lock:
            stats = self._aggregate_stats

            if stats.total_trades < 5:
                return [("Insufficient data", 0.0)]

            priorities = []

            # Exit quality impact
            poor_exit_rate = (stats.poor_exits + stats.stopped_out_exits) / stats.total_trades
            if poor_exit_rate > 0.3:
                priorities.append(
                    ("Exit timing", poor_exit_rate * (1.0 - stats.avg_capture_ratio))
                )

            # Entry quality impact
            poor_entry_rate = (stats.poor_entries + stats.acceptable_entries) / stats.total_trades
            if poor_entry_rate > 0.4:
                priorities.append(("Entry timing", poor_entry_rate * 0.5))

            # Stop quality impact
            poor_stop_rate = (stats.tight_stops + stats.loose_stops + stats.no_stops) / stats.total_trades
            if poor_stop_rate > 0.5:
                priorities.append(("Stop placement", poor_stop_rate * stats.avg_pain_ratio))

            # Holding period optimization
            if stats.avg_holding_minutes > stats.avg_optimal_holding_minutes * 1.5:
                hold_excess = (
                    stats.avg_holding_minutes - stats.avg_optimal_holding_minutes
                ) / stats.avg_optimal_holding_minutes
                priorities.append(("Holding period", min(1.0, hold_excess * 0.3)))

            # Pain ratio (drawdown during trades)
            if stats.avg_pain_ratio > 0.5:
                priorities.append(("Drawdown management", stats.avg_pain_ratio * 0.4))

            # Sort by impact
            priorities.sort(key=lambda x: x[1], reverse=True)

            return priorities if priorities else [("No major issues detected", 0.0)]

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive forensics status."""
        with self._lock:
            stats = self._aggregate_stats

            return {
                "total_trades_analyzed": stats.total_trades,
                "efficiency_metrics": {
                    "avg_capture_ratio": stats.avg_capture_ratio,
                    "avg_pain_ratio": stats.avg_pain_ratio,
                    "avg_edge_efficiency": stats.avg_edge_efficiency,
                },
                "exit_quality_distribution": {
                    "optimal": stats.optimal_exits,
                    "good": stats.good_exits,
                    "acceptable": stats.acceptable_exits,
                    "poor": stats.poor_exits,
                    "stopped_out": stats.stopped_out_exits,
                },
                "entry_quality_distribution": {
                    "optimal": stats.optimal_entries,
                    "good": stats.good_entries,
                    "acceptable": stats.acceptable_entries,
                    "poor": stats.poor_entries,
                },
                "stop_quality_distribution": {
                    "too_tight": stats.tight_stops,
                    "optimal": stats.optimal_stops,
                    "too_loose": stats.loose_stops,
                    "not_used": stats.no_stops,
                },
                "timing_stats": {
                    "avg_mae_minutes": stats.avg_mae_minutes,
                    "avg_mfe_minutes": stats.avg_mfe_minutes,
                    "avg_holding_minutes": stats.avg_holding_minutes,
                    "avg_optimal_holding_minutes": stats.avg_optimal_holding_minutes,
                },
                "improvement_priorities": self.get_improvement_priorities(),
            }


# Singleton accessor
_instance: Optional[TradeForensics] = None


def get_trade_forensics(config: Optional[ForensicsConfig] = None) -> TradeForensics:
    """Get or create the singleton TradeForensics instance."""
    global _instance
    if _instance is None:
        _instance = TradeForensics(config=config)
    return _instance


def reset_trade_forensics() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    if _instance is not None:
        _instance._initialized = False
    _instance = None
    TradeForensics._instance = None
