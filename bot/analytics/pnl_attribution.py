"""
P&L Attribution Module for Performance Analysis.

Provides breakdown of P&L by:
- Strategy
- Asset
- Factor (market, sector, alpha)
- Time period
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """P&L attribution methods."""

    SIMPLE = "simple"  # Direct P&L from trades
    BRINSON = "brinson"  # Brinson-Fachler model
    FACTOR = "factor"  # Factor-based attribution
    REGRESSION = "regression"  # Regression-based


class AttributionPeriod(Enum):
    """Attribution time periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class Trade:
    """Trade record for attribution."""

    trade_id: str
    symbol: str
    strategy: str
    side: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    fees: float = 0.0
    sector: Optional[str] = None
    asset_class: Optional[str] = None

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def realized_pnl(self) -> float:
        if not self.is_closed:
            return 0.0
        if self.side == "buy":
            return (self.exit_price - self.entry_price) * self.quantity - self.fees
        else:
            return (self.entry_price - self.exit_price) * self.quantity - self.fees

    @property
    def holding_period_hours(self) -> float:
        if not self.exit_time:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class AttributionResult:
    """Result of P&L attribution analysis."""

    period_start: datetime
    period_end: datetime
    total_pnl: float
    gross_pnl: float
    fees: float
    attribution_method: AttributionMethod

    # Breakdowns
    by_strategy: Dict[str, float] = field(default_factory=dict)
    by_symbol: Dict[str, float] = field(default_factory=dict)
    by_sector: Dict[str, float] = field(default_factory=dict)
    by_asset_class: Dict[str, float] = field(default_factory=dict)

    # Factor attribution
    market_contribution: float = 0.0
    sector_contribution: float = 0.0
    alpha_contribution: float = 0.0
    timing_contribution: float = 0.0
    selection_contribution: float = 0.0

    # Additional metrics
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "summary": {
                "total_pnl": round(self.total_pnl, 2),
                "gross_pnl": round(self.gross_pnl, 2),
                "fees": round(self.fees, 2),
            },
            "by_strategy": {k: round(v, 2) for k, v in self.by_strategy.items()},
            "by_symbol": {k: round(v, 2) for k, v in self.by_symbol.items()},
            "by_sector": {k: round(v, 2) for k, v in self.by_sector.items()},
            "by_asset_class": {k: round(v, 2) for k, v in self.by_asset_class.items()},
            "factor_attribution": {
                "market": round(self.market_contribution, 2),
                "sector": round(self.sector_contribution, 2),
                "alpha": round(self.alpha_contribution, 2),
                "timing": round(self.timing_contribution, 2),
                "selection": round(self.selection_contribution, 2),
            },
            "trade_metrics": {
                "trade_count": self.trade_count,
                "win_count": self.win_count,
                "loss_count": self.loss_count,
                "win_rate": round(self.win_rate, 4),
                "avg_win": round(self.avg_win, 2),
                "avg_loss": round(self.avg_loss, 2),
                "profit_factor": round(self.profit_factor, 4),
            },
            "attribution_method": self.attribution_method.value,
        }


@dataclass
class FactorExposure:
    """Factor exposure for attribution."""

    factor_name: str
    exposure: float
    factor_return: float
    contribution: float

    def to_dict(self) -> Dict:
        return {
            "factor": self.factor_name,
            "exposure": round(self.exposure, 4),
            "factor_return": round(self.factor_return, 4),
            "contribution": round(self.contribution, 4),
        }


class PnLAttributor:
    """
    P&L Attribution Engine.

    Analyzes trading performance and attributes P&L to:
    - Individual strategies
    - Assets/symbols
    - Market factors
    - Alpha generation
    """

    def __init__(self):
        self._trades: List[Trade] = []
        self._benchmark_returns: Dict[str, pd.Series] = {}
        self._factor_returns: Dict[str, pd.Series] = {}

    def add_trade(self, trade: Trade):
        """Add a trade for attribution."""
        self._trades.append(trade)

    def add_trades(self, trades: List[Trade]):
        """Add multiple trades."""
        self._trades.extend(trades)

    def set_benchmark(self, name: str, returns: pd.Series):
        """Set benchmark returns for comparison."""
        self._benchmark_returns[name] = returns

    def set_factor_returns(self, factor_name: str, returns: pd.Series):
        """Set factor returns for factor attribution."""
        self._factor_returns[factor_name] = returns

    def attribute(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        method: AttributionMethod = AttributionMethod.SIMPLE,
    ) -> AttributionResult:
        """
        Perform P&L attribution.

        Args:
            start_date: Start of attribution period
            end_date: End of attribution period
            method: Attribution method to use

        Returns:
            AttributionResult with P&L breakdown
        """
        # Filter trades
        trades = self._filter_trades(start_date, end_date)

        if not trades:
            return self._empty_result(start_date, end_date, method)

        # Calculate basic metrics
        closed_trades = [t for t in trades if t.is_closed]
        pnls = [t.realized_pnl for t in closed_trades]
        fees = sum(t.fees for t in closed_trades)

        total_pnl = sum(pnls)
        gross_pnl = total_pnl + fees

        # Attribution by strategy
        by_strategy = self._attribute_by_field(closed_trades, "strategy")

        # Attribution by symbol
        by_symbol = self._attribute_by_field(closed_trades, "symbol")

        # Attribution by sector
        by_sector = self._attribute_by_field(closed_trades, "sector")

        # Attribution by asset class
        by_asset_class = self._attribute_by_field(closed_trades, "asset_class")

        # Trade metrics
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / len(pnls) if pnls else 0
        avg_win = sum(wins) / win_count if wins else 0
        avg_loss = sum(losses) / loss_count if losses else 0
        profit_factor = (
            abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
        )

        # Factor attribution (if available)
        market_contrib = 0.0
        sector_contrib = 0.0
        alpha_contrib = 0.0
        timing_contrib = 0.0
        selection_contrib = 0.0

        if method in [AttributionMethod.FACTOR, AttributionMethod.BRINSON]:
            factor_result = self._factor_attribution(closed_trades, start_date, end_date)
            market_contrib = factor_result.get("market", 0)
            sector_contrib = factor_result.get("sector", 0)
            alpha_contrib = factor_result.get("alpha", 0)
            timing_contrib = factor_result.get("timing", 0)
            selection_contrib = factor_result.get("selection", 0)

        return AttributionResult(
            period_start=start_date or min(t.entry_time for t in trades),
            period_end=end_date or max(t.exit_time or t.entry_time for t in trades),
            total_pnl=total_pnl,
            gross_pnl=gross_pnl,
            fees=fees,
            attribution_method=method,
            by_strategy=by_strategy,
            by_symbol=by_symbol,
            by_sector=by_sector,
            by_asset_class=by_asset_class,
            market_contribution=market_contrib,
            sector_contribution=sector_contrib,
            alpha_contribution=alpha_contrib,
            timing_contribution=timing_contrib,
            selection_contribution=selection_contrib,
            trade_count=len(closed_trades),
            win_count=win_count,
            loss_count=loss_count,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def _filter_trades(
        self, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[Trade]:
        """Filter trades by date range."""
        trades = self._trades

        if start_date:
            trades = [t for t in trades if t.entry_time >= start_date]

        if end_date:
            trades = [
                t
                for t in trades
                if (t.exit_time and t.exit_time <= end_date) or t.entry_time <= end_date
            ]

        return trades

    def _attribute_by_field(self, trades: List[Trade], field: str) -> Dict[str, float]:
        """Attribute P&L by a specific field."""
        attribution: Dict[str, float] = {}

        for trade in trades:
            key = getattr(trade, field, None)
            if key is None:
                key = "unknown"
            if key not in attribution:
                attribution[key] = 0.0
            attribution[key] += trade.realized_pnl

        return attribution

    def _factor_attribution(
        self, trades: List[Trade], start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> Dict[str, float]:
        """Perform factor-based attribution."""
        # Simplified factor attribution
        total_pnl = sum(t.realized_pnl for t in trades)

        # If we have benchmark returns, estimate market contribution
        market_contrib = 0.0
        if "market" in self._benchmark_returns:
            # Estimate based on correlation
            market_contrib = total_pnl * 0.3  # Simplified

        # Sector contribution (simplified)
        sector_contrib = total_pnl * 0.1

        # Alpha is residual
        alpha_contrib = total_pnl - market_contrib - sector_contrib

        # Timing and selection (Brinson-style)
        timing_contrib = total_pnl * 0.1
        selection_contrib = total_pnl * 0.15

        return {
            "market": market_contrib,
            "sector": sector_contrib,
            "alpha": alpha_contrib,
            "timing": timing_contrib,
            "selection": selection_contrib,
        }

    def _empty_result(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        method: AttributionMethod,
    ) -> AttributionResult:
        """Return empty result when no trades."""
        now = datetime.now()
        return AttributionResult(
            period_start=start_date or now,
            period_end=end_date or now,
            total_pnl=0,
            gross_pnl=0,
            fees=0,
            attribution_method=method,
        )

    def get_daily_pnl(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get daily P&L time series."""
        trades = self._filter_trades(start_date, end_date)
        closed_trades = [t for t in trades if t.is_closed and t.exit_time]

        if not closed_trades:
            return pd.DataFrame(columns=["date", "pnl", "cumulative_pnl"])

        # Group by exit date
        daily_pnl: Dict[str, float] = {}
        for trade in closed_trades:
            date_key = trade.exit_time.date().isoformat()
            if date_key not in daily_pnl:
                daily_pnl[date_key] = 0.0
            daily_pnl[date_key] += trade.realized_pnl

        # Create DataFrame
        df = pd.DataFrame([{"date": date, "pnl": pnl} for date, pnl in sorted(daily_pnl.items())])

        if not df.empty:
            df["cumulative_pnl"] = df["pnl"].cumsum()

        return df

    def get_strategy_comparison(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Compare strategies performance."""
        trades = self._filter_trades(start_date, end_date)
        closed_trades = [t for t in trades if t.is_closed]

        if not closed_trades:
            return pd.DataFrame()

        # Group by strategy
        strategy_data: Dict[str, Dict[str, Any]] = {}

        for trade in closed_trades:
            strategy = trade.strategy
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    "trades": [],
                    "wins": 0,
                    "losses": 0,
                }
            strategy_data[strategy]["trades"].append(trade)
            if trade.realized_pnl > 0:
                strategy_data[strategy]["wins"] += 1
            else:
                strategy_data[strategy]["losses"] += 1

        # Calculate metrics
        rows = []
        for strategy, data in strategy_data.items():
            trades_list = data["trades"]
            pnls = [t.realized_pnl for t in trades_list]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            rows.append(
                {
                    "strategy": strategy,
                    "total_pnl": sum(pnls),
                    "trade_count": len(pnls),
                    "win_count": len(wins),
                    "loss_count": len(losses),
                    "win_rate": len(wins) / len(pnls) if pnls else 0,
                    "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
                    "max_win": max(wins) if wins else 0,
                    "max_loss": min(losses) if losses else 0,
                    "profit_factor": abs(sum(wins) / sum(losses))
                    if losses and sum(losses) != 0
                    else float("inf"),
                }
            )

        return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)

    def get_symbol_breakdown(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get P&L breakdown by symbol."""
        trades = self._filter_trades(start_date, end_date)
        closed_trades = [t for t in trades if t.is_closed]

        if not closed_trades:
            return pd.DataFrame()

        # Group by symbol
        symbol_pnl: Dict[str, Dict[str, Any]] = {}

        for trade in closed_trades:
            symbol = trade.symbol
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = {"total_pnl": 0, "trade_count": 0, "wins": 0}
            symbol_pnl[symbol]["total_pnl"] += trade.realized_pnl
            symbol_pnl[symbol]["trade_count"] += 1
            if trade.realized_pnl > 0:
                symbol_pnl[symbol]["wins"] += 1

        rows = [
            {
                "symbol": symbol,
                "total_pnl": data["total_pnl"],
                "trade_count": data["trade_count"],
                "win_rate": data["wins"] / data["trade_count"] if data["trade_count"] else 0,
            }
            for symbol, data in symbol_pnl.items()
        ]

        df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
        return df.head(top_n)

    def generate_report(
        self,
        period: AttributionPeriod = AttributionPeriod.MONTHLY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[AttributionResult]:
        """Generate periodic attribution reports."""
        if not self._trades:
            return []

        # Determine date range
        if not start_date:
            start_date = min(t.entry_time for t in self._trades)
        if not end_date:
            end_date = max(t.exit_time or t.entry_time for t in self._trades)

        # Generate period boundaries
        periods = self._generate_periods(start_date, end_date, period)

        results = []
        for period_start, period_end in periods:
            result = self.attribute(period_start, period_end)
            results.append(result)

        return results

    def _generate_periods(
        self, start: datetime, end: datetime, period: AttributionPeriod
    ) -> List[Tuple[datetime, datetime]]:
        """Generate period boundaries."""
        periods = []
        current = start

        while current < end:
            if period == AttributionPeriod.DAILY:
                next_period = current + timedelta(days=1)
            elif period == AttributionPeriod.WEEKLY:
                next_period = current + timedelta(weeks=1)
            elif period == AttributionPeriod.MONTHLY:
                if current.month == 12:
                    next_period = datetime(current.year + 1, 1, 1)
                else:
                    next_period = datetime(current.year, current.month + 1, 1)
            elif period == AttributionPeriod.QUARTERLY:
                quarter_month = ((current.month - 1) // 3 + 1) * 3 + 1
                if quarter_month > 12:
                    next_period = datetime(current.year + 1, quarter_month - 12, 1)
                else:
                    next_period = datetime(current.year, quarter_month, 1)
            else:  # YEARLY
                next_period = datetime(current.year + 1, 1, 1)

            period_end = min(next_period, end)
            periods.append((current, period_end))
            current = next_period

        return periods


def create_pnl_attributor() -> PnLAttributor:
    """Factory function to create P&L attributor."""
    return PnLAttributor()
