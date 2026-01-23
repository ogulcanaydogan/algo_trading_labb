"""
Slippage Tracker - Measure execution quality.

Tracks the difference between expected and actual execution prices
to monitor broker quality and optimize order timing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SlippageRecord:
    """Record of a single execution's slippage."""

    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    expected_price: float
    executed_price: float
    quantity: float
    timestamp: datetime = field(default_factory=datetime.now)
    market_volatility: Optional[float] = None
    order_type: str = "market"
    exchange: Optional[str] = None

    @property
    def slippage_absolute(self) -> float:
        """Absolute slippage in price units."""
        return self.executed_price - self.expected_price

    @property
    def slippage_percent(self) -> float:
        """Slippage as percentage of expected price."""
        if self.expected_price == 0:
            return 0.0
        return (self.slippage_absolute / self.expected_price) * 100

    @property
    def slippage_bps(self) -> float:
        """Slippage in basis points."""
        return self.slippage_percent * 100

    @property
    def cost_usd(self) -> float:
        """Total slippage cost in USD."""
        return abs(self.slippage_absolute) * self.quantity

    @property
    def is_favorable(self) -> bool:
        """Check if slippage was in our favor."""
        if self.side == "buy":
            return self.executed_price < self.expected_price
        else:
            return self.executed_price > self.expected_price

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "expected_price": self.expected_price,
            "executed_price": self.executed_price,
            "quantity": self.quantity,
            "slippage_percent": round(self.slippage_percent, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "cost_usd": round(self.cost_usd, 2),
            "is_favorable": self.is_favorable,
            "timestamp": self.timestamp.isoformat(),
            "market_volatility": self.market_volatility,
            "order_type": self.order_type,
            "exchange": self.exchange,
        }


@dataclass
class SlippageStats:
    """Aggregated slippage statistics."""

    total_trades: int = 0
    total_slippage_usd: float = 0.0
    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    favorable_rate: float = 0.0
    by_symbol: Dict[str, Dict] = field(default_factory=dict)
    by_exchange: Dict[str, Dict] = field(default_factory=dict)
    by_hour: Dict[int, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "total_slippage_usd": round(self.total_slippage_usd, 2),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "max_slippage_bps": round(self.max_slippage_bps, 2),
            "favorable_rate": round(self.favorable_rate, 4),
            "by_symbol": self.by_symbol,
            "by_exchange": self.by_exchange,
            "by_hour": self.by_hour,
        }


class SlippageTracker:
    """
    Track and analyze execution slippage.

    Features:
    - Record expected vs actual prices
    - Calculate slippage statistics by symbol, exchange, time
    - Identify patterns (best/worst execution times)
    - Alert on excessive slippage
    - Persist history for analysis
    """

    def __init__(
        self,
        data_dir: str = "data/execution",
        alert_threshold_bps: float = 50.0,  # Alert if slippage > 50 bps
        history_days: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alert_threshold_bps = alert_threshold_bps
        self.history_days = history_days

        self.records: List[SlippageRecord] = []
        self._load_history()

    def _load_history(self):
        """Load historical slippage records."""
        history_file = self.data_dir / "slippage_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    cutoff = datetime.now() - timedelta(days=self.history_days)
                    for record in data:
                        ts = datetime.fromisoformat(record["timestamp"])
                        if ts > cutoff:
                            self.records.append(
                                SlippageRecord(
                                    order_id=record["order_id"],
                                    symbol=record["symbol"],
                                    side=record["side"],
                                    expected_price=record["expected_price"],
                                    executed_price=record["executed_price"],
                                    quantity=record["quantity"],
                                    timestamp=ts,
                                    market_volatility=record.get("market_volatility"),
                                    order_type=record.get("order_type", "market"),
                                    exchange=record.get("exchange"),
                                )
                            )
                logger.info(f"Loaded {len(self.records)} slippage records")
            except Exception as e:
                logger.error(f"Error loading slippage history: {e}")

    def _save_history(self):
        """Save slippage records to disk."""
        history_file = self.data_dir / "slippage_history.json"
        try:
            data = [r.to_dict() for r in self.records]
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving slippage history: {e}")

    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: Literal["buy", "sell"],
        expected_price: float,
        executed_price: float,
        quantity: float,
        market_volatility: Optional[float] = None,
        order_type: str = "market",
        exchange: Optional[str] = None,
    ) -> SlippageRecord:
        """
        Record an execution and its slippage.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            expected_price: Price when order was placed
            executed_price: Actual fill price
            quantity: Order quantity
            market_volatility: Current market volatility (optional)
            order_type: Order type (market, limit, etc.)
            exchange: Exchange name

        Returns:
            SlippageRecord with calculated metrics
        """
        record = SlippageRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            executed_price=executed_price,
            quantity=quantity,
            market_volatility=market_volatility,
            order_type=order_type,
            exchange=exchange,
        )

        self.records.append(record)
        self._save_history()

        # Log and alert
        if abs(record.slippage_bps) > self.alert_threshold_bps:
            logger.warning(
                f"HIGH SLIPPAGE ALERT: {symbol} {side} "
                f"{record.slippage_bps:.1f} bps (${record.cost_usd:.2f})"
            )
        else:
            logger.debug(f"Slippage recorded: {symbol} {side} {record.slippage_bps:.1f} bps")

        return record

    def get_stats(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        days: Optional[int] = None,
    ) -> SlippageStats:
        """
        Calculate slippage statistics.

        Args:
            symbol: Filter by symbol (optional)
            exchange: Filter by exchange (optional)
            days: Look back period in days (optional)

        Returns:
            SlippageStats with aggregated metrics
        """
        # Filter records
        records = self.records
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            records = [r for r in records if r.timestamp > cutoff]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        if exchange:
            records = [r for r in records if r.exchange == exchange]

        if not records:
            return SlippageStats()

        # Calculate stats
        slippages_bps = [r.slippage_bps for r in records]
        costs = [r.cost_usd for r in records]
        favorable = [r.is_favorable for r in records]

        stats = SlippageStats(
            total_trades=len(records),
            total_slippage_usd=sum(costs),
            avg_slippage_bps=np.mean(np.abs(slippages_bps)),
            max_slippage_bps=max(np.abs(slippages_bps)),
            favorable_rate=sum(favorable) / len(favorable),
        )

        # By symbol
        by_symbol = defaultdict(list)
        for r in records:
            by_symbol[r.symbol].append(r)
        for sym, recs in by_symbol.items():
            stats.by_symbol[sym] = {
                "trades": len(recs),
                "avg_slippage_bps": round(np.mean([abs(r.slippage_bps) for r in recs]), 2),
                "total_cost_usd": round(sum(r.cost_usd for r in recs), 2),
            }

        # By exchange
        by_exchange = defaultdict(list)
        for r in records:
            if r.exchange:
                by_exchange[r.exchange].append(r)
        for exch, recs in by_exchange.items():
            stats.by_exchange[exch] = {
                "trades": len(recs),
                "avg_slippage_bps": round(np.mean([abs(r.slippage_bps) for r in recs]), 2),
            }

        # By hour (find best/worst times)
        by_hour = defaultdict(list)
        for r in records:
            by_hour[r.timestamp.hour].append(r)
        for hour, recs in by_hour.items():
            stats.by_hour[hour] = {
                "trades": len(recs),
                "avg_slippage_bps": round(np.mean([abs(r.slippage_bps) for r in recs]), 2),
            }

        return stats

    def get_best_execution_hours(self, top_n: int = 3) -> List[int]:
        """Get hours with lowest average slippage."""
        stats = self.get_stats()
        if not stats.by_hour:
            return list(range(top_n))

        sorted_hours = sorted(stats.by_hour.items(), key=lambda x: x[1]["avg_slippage_bps"])
        return [h for h, _ in sorted_hours[:top_n]]

    def get_worst_execution_hours(self, top_n: int = 3) -> List[int]:
        """Get hours with highest average slippage."""
        stats = self.get_stats()
        if not stats.by_hour:
            return []

        sorted_hours = sorted(
            stats.by_hour.items(), key=lambda x: x[1]["avg_slippage_bps"], reverse=True
        )
        return [h for h, _ in sorted_hours[:top_n]]

    def estimate_slippage(
        self,
        symbol: str,
        quantity: float,
        current_volatility: Optional[float] = None,
    ) -> float:
        """
        Estimate expected slippage for a trade.

        Based on historical data for similar conditions.

        Returns:
            Estimated slippage in basis points
        """
        # Get historical records for this symbol
        symbol_records = [r for r in self.records if r.symbol == symbol]

        if not symbol_records:
            # No history, use conservative estimate
            return 10.0  # 10 bps default

        # Calculate average slippage
        avg_slippage = np.mean([abs(r.slippage_bps) for r in symbol_records])

        # Adjust for quantity (larger orders typically have more slippage)
        avg_quantity = np.mean([r.quantity for r in symbol_records])
        quantity_factor = (quantity / avg_quantity) ** 0.5 if avg_quantity > 0 else 1.0

        # Adjust for volatility if provided
        volatility_factor = 1.0
        if current_volatility:
            vol_records = [r for r in symbol_records if r.market_volatility]
            if vol_records:
                avg_vol = np.mean([r.market_volatility for r in vol_records])
                volatility_factor = current_volatility / avg_vol if avg_vol > 0 else 1.0

        estimated = avg_slippage * quantity_factor * volatility_factor
        return min(estimated, 100.0)  # Cap at 100 bps

    def get_summary_report(self) -> Dict:
        """Generate a summary report of execution quality."""
        stats_7d = self.get_stats(days=7)
        stats_30d = self.get_stats(days=30)

        return {
            "period": {
                "7_day": stats_7d.to_dict(),
                "30_day": stats_30d.to_dict(),
            },
            "best_hours": self.get_best_execution_hours(),
            "worst_hours": self.get_worst_execution_hours(),
            "recommendations": self._generate_recommendations(stats_30d),
        }

    def _generate_recommendations(self, stats: SlippageStats) -> List[str]:
        """Generate recommendations based on slippage analysis."""
        recommendations = []

        if stats.avg_slippage_bps > 20:
            recommendations.append(
                f"High average slippage ({stats.avg_slippage_bps:.1f} bps). "
                "Consider using limit orders or reducing order sizes."
            )

        if stats.favorable_rate < 0.4:
            recommendations.append(
                f"Low favorable slippage rate ({stats.favorable_rate:.1%}). "
                "Review order timing and execution strategy."
            )

        best_hours = self.get_best_execution_hours()
        if best_hours:
            recommendations.append(
                f"Best execution hours (UTC): {best_hours}. "
                "Consider scheduling trades during these times."
            )

        # Symbol-specific recommendations
        for symbol, data in stats.by_symbol.items():
            if data["avg_slippage_bps"] > 30:
                recommendations.append(
                    f"{symbol}: High slippage ({data['avg_slippage_bps']:.1f} bps). "
                    "Consider splitting large orders or using TWAP."
                )

        return recommendations


# Singleton instance
_tracker: Optional[SlippageTracker] = None


def get_slippage_tracker(data_dir: str = "data/execution") -> SlippageTracker:
    """Get or create the slippage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SlippageTracker(data_dir=data_dir)
    return _tracker
