"""
Performance Attribution Tracking.

Analyzes and attributes P&L to different factors:
- Strategy contribution
- Model contribution
- Asset allocation
- Timing decisions
- Factor exposures
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttributionFactor(Enum):
    """Types of attribution factors."""
    STRATEGY = "strategy"
    MODEL = "model"
    ASSET = "asset"
    TIMING = "timing"
    MARKET_TYPE = "market_type"
    REGIME = "regime"
    SENTIMENT = "sentiment"


@dataclass
class TradeRecord:
    """Record of a single trade for attribution."""
    trade_id: str
    symbol: str
    market_type: str
    action: str  # BUY, SELL
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    pnl_pct: float
    strategy: str
    model: str
    regime: str
    confidence: float
    sentiment_score: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    holding_period_hours: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "strategy": self.strategy,
            "model": self.model,
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "sentiment_score": round(self.sentiment_score, 4) if self.sentiment_score else None,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "holding_period_hours": round(self.holding_period_hours, 2),
        }


@dataclass
class AttributionResult:
    """Attribution analysis result for a specific factor."""
    factor: AttributionFactor
    category: str  # e.g., strategy name, model name
    total_pnl: float
    pnl_contribution_pct: float  # % of total P&L
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_period: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor.value,
            "category": self.category,
            "total_pnl": round(self.total_pnl, 2),
            "pnl_contribution_pct": round(self.pnl_contribution_pct, 2),
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_holding_period": round(self.avg_holding_period, 2),
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance attribution report."""
    period_start: datetime
    period_end: datetime
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    overall_win_rate: float
    overall_sharpe: float
    attributions_by_strategy: List[AttributionResult]
    attributions_by_model: List[AttributionResult]
    attributions_by_asset: List[AttributionResult]
    attributions_by_market: List[AttributionResult]
    attributions_by_regime: List[AttributionResult]
    top_performers: List[Dict[str, Any]]
    worst_performers: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "total_trades": self.total_trades,
            "overall_win_rate": round(self.overall_win_rate, 4),
            "overall_sharpe": round(self.overall_sharpe, 2),
            "attributions_by_strategy": [a.to_dict() for a in self.attributions_by_strategy],
            "attributions_by_model": [a.to_dict() for a in self.attributions_by_model],
            "attributions_by_asset": [a.to_dict() for a in self.attributions_by_asset],
            "attributions_by_market": [a.to_dict() for a in self.attributions_by_market],
            "attributions_by_regime": [a.to_dict() for a in self.attributions_by_regime],
            "top_performers": self.top_performers,
            "worst_performers": self.worst_performers,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


class PerformanceAttributor:
    """
    Tracks and attributes trading performance.

    Features:
    - Trade logging with full context
    - Multi-factor attribution analysis
    - Strategy/model performance comparison
    - Timing and regime analysis
    - Recommendations generation

    Usage:
        attributor = PerformanceAttributor()

        # Log a trade
        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=52000,
            strategy="momentum",
            model="lstm",
        )

        # Get attribution report
        report = attributor.generate_report(days=30)
    """

    def __init__(
        self,
        data_dir: str = "data/performance_attribution",
        initial_capital: float = 10000,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.initial_capital = initial_capital
        self.trades: List[TradeRecord] = []

        self._load_trades()

    def _get_trades_file(self) -> Path:
        return self.data_dir / "trade_history.json"

    def _load_trades(self) -> None:
        """Load trade history from disk."""
        trades_file = self._get_trades_file()
        if trades_file.exists():
            try:
                with open(trades_file, "r") as f:
                    data = json.load(f)
                    for t in data:
                        t["entry_time"] = datetime.fromisoformat(t["entry_time"])
                        if t.get("exit_time"):
                            t["exit_time"] = datetime.fromisoformat(t["exit_time"])
                        self.trades.append(TradeRecord(**t))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load trade history: {e}")

    def _save_trades(self) -> None:
        """Save trade history to disk."""
        data = [t.to_dict() for t in self.trades]
        with open(self._get_trades_file(), "w") as f:
            json.dump(data, f, indent=2)

    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        action: str,
        quantity: float,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        strategy: str = "default",
        model: str = "default",
        regime: str = "unknown",
        confidence: float = 0.5,
        sentiment_score: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        market_type: str = "crypto",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        """
        Log a trade for attribution.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            action: BUY or SELL
            quantity: Position size
            entry_price: Entry price
            exit_price: Exit price (if closed)
            pnl: P&L (calculated if not provided)
            strategy: Strategy name
            model: Model used for signal
            regime: Market regime at entry
            confidence: Signal confidence
            sentiment_score: News sentiment at entry
            entry_time: Entry timestamp
            exit_time: Exit timestamp (if closed)
            market_type: crypto, stock, commodity
            metadata: Additional context

        Returns:
            TradeRecord object
        """
        entry_time = entry_time or datetime.now()

        # Calculate P&L if not provided
        if pnl is None and exit_price is not None:
            if action == "BUY":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

        pnl = pnl or 0.0
        pnl_pct = pnl / (entry_price * quantity) if entry_price * quantity > 0 else 0

        # Calculate holding period
        if exit_time:
            holding_hours = (exit_time - entry_time).total_seconds() / 3600
        else:
            holding_hours = 0

        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            market_type=market_type,
            action=action,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=strategy,
            model=model,
            regime=regime,
            confidence=confidence,
            sentiment_score=sentiment_score,
            entry_time=entry_time,
            exit_time=exit_time,
            holding_period_hours=holding_hours,
            metadata=metadata or {},
        )

        self.trades.append(trade)
        self._save_trades()

        return trade

    def update_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
    ) -> Optional[TradeRecord]:
        """Update a trade with exit information."""
        exit_time = exit_time or datetime.now()

        for trade in self.trades:
            if trade.trade_id == trade_id:
                trade.exit_price = exit_price
                trade.exit_time = exit_time

                # Recalculate P&L
                if trade.action == "BUY":
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - exit_price) * trade.quantity

                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                trade.holding_period_hours = (exit_time - trade.entry_time).total_seconds() / 3600

                self._save_trades()
                return trade

        return None

    def _calculate_attribution(
        self,
        trades: List[TradeRecord],
        factor: AttributionFactor,
        groupby_func,
        total_pnl: float,
    ) -> List[AttributionResult]:
        """Calculate attribution for a specific factor."""
        groups: Dict[str, List[TradeRecord]] = {}

        for trade in trades:
            key = groupby_func(trade)
            if key not in groups:
                groups[key] = []
            groups[key].append(trade)

        results = []

        for category, group_trades in groups.items():
            pnls = [t.pnl for t in group_trades]
            returns = [t.pnl_pct for t in group_trades]
            holding_periods = [t.holding_period_hours for t in group_trades]

            group_pnl = sum(pnls)
            contribution = (group_pnl / total_pnl * 100) if total_pnl != 0 else 0

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

            # Sharpe ratio
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized for hourly
            else:
                sharpe = 0

            # Max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_dd = max(drawdowns) / self.initial_capital if len(drawdowns) > 0 else 0

            results.append(AttributionResult(
                factor=factor,
                category=category,
                total_pnl=group_pnl,
                pnl_contribution_pct=contribution,
                trade_count=len(group_trades),
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=min(profit_factor, 100),  # Cap at 100
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                avg_holding_period=np.mean(holding_periods) if holding_periods else 0,
            ))

        # Sort by contribution
        results.sort(key=lambda x: x.total_pnl, reverse=True)

        return results

    def generate_report(
        self,
        days: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceReport:
        """
        Generate comprehensive performance attribution report.

        Args:
            days: Number of days to analyze (if start/end not specified)
            start_date: Start of period
            end_date: End of period

        Returns:
            PerformanceReport with full attribution analysis
        """
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=days))

        # Filter trades by period
        period_trades = [
            t for t in self.trades
            if start_date <= t.entry_time <= end_date
        ]

        # Calculate overall metrics
        total_pnl = sum(t.pnl for t in period_trades)
        total_pnl_pct = total_pnl / self.initial_capital

        if period_trades:
            wins = [t for t in period_trades if t.pnl > 0]
            overall_win_rate = len(wins) / len(period_trades)

            returns = [t.pnl_pct for t in period_trades]
            overall_sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
                           if np.std(returns) > 0 else 0)
        else:
            overall_win_rate = 0
            overall_sharpe = 0

        # Calculate attributions
        attributions_by_strategy = self._calculate_attribution(
            period_trades, AttributionFactor.STRATEGY,
            lambda t: t.strategy, total_pnl
        )

        attributions_by_model = self._calculate_attribution(
            period_trades, AttributionFactor.MODEL,
            lambda t: t.model, total_pnl
        )

        attributions_by_asset = self._calculate_attribution(
            period_trades, AttributionFactor.ASSET,
            lambda t: t.symbol, total_pnl
        )

        attributions_by_market = self._calculate_attribution(
            period_trades, AttributionFactor.MARKET_TYPE,
            lambda t: t.market_type, total_pnl
        )

        attributions_by_regime = self._calculate_attribution(
            period_trades, AttributionFactor.REGIME,
            lambda t: t.regime, total_pnl
        )

        # Top and worst performers
        sorted_trades = sorted(period_trades, key=lambda t: t.pnl, reverse=True)
        top_performers = [t.to_dict() for t in sorted_trades[:5]]
        worst_performers = [t.to_dict() for t in sorted_trades[-5:]]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            attributions_by_strategy,
            attributions_by_model,
            attributions_by_regime,
            overall_win_rate,
        )

        return PerformanceReport(
            period_start=start_date,
            period_end=end_date,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=len(period_trades),
            overall_win_rate=overall_win_rate,
            overall_sharpe=overall_sharpe,
            attributions_by_strategy=attributions_by_strategy,
            attributions_by_model=attributions_by_model,
            attributions_by_asset=attributions_by_asset,
            attributions_by_market=attributions_by_market,
            attributions_by_regime=attributions_by_regime,
            top_performers=top_performers,
            worst_performers=worst_performers,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        strategy_attr: List[AttributionResult],
        model_attr: List[AttributionResult],
        regime_attr: List[AttributionResult],
        overall_win_rate: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Strategy recommendations
        for attr in strategy_attr:
            if attr.total_pnl < 0 and attr.trade_count >= 5:
                recommendations.append(
                    f"Consider reducing allocation to {attr.category} strategy "
                    f"(P&L: ${attr.total_pnl:.2f}, Win Rate: {attr.win_rate:.1%})"
                )
            elif attr.profit_factor > 2 and attr.trade_count >= 5:
                recommendations.append(
                    f"Consider increasing allocation to {attr.category} strategy "
                    f"(Profit Factor: {attr.profit_factor:.1f})"
                )

        # Model recommendations
        for attr in model_attr:
            if attr.win_rate < 0.4 and attr.trade_count >= 10:
                recommendations.append(
                    f"{attr.category} model underperforming "
                    f"(Win Rate: {attr.win_rate:.1%}) - consider retraining"
                )

        # Regime recommendations
        for attr in regime_attr:
            if attr.total_pnl < 0:
                recommendations.append(
                    f"Poor performance in {attr.category} regime - "
                    f"consider reducing position sizes during this regime"
                )

        # Overall recommendations
        if overall_win_rate < 0.45:
            recommendations.append(
                "Overall win rate below 45% - review signal quality and entry criteria"
            )

        return recommendations[:5]  # Top 5 recommendations

    def get_daily_summary(
        self,
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Get daily performance summary."""
        target = target_date or date.today()

        day_trades = [
            t for t in self.trades
            if t.entry_time.date() == target
        ]

        if not day_trades:
            return {
                "date": target.isoformat(),
                "trade_count": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "best_trade": None,
                "worst_trade": None,
            }

        total_pnl = sum(t.pnl for t in day_trades)
        wins = [t for t in day_trades if t.pnl > 0]

        sorted_trades = sorted(day_trades, key=lambda t: t.pnl, reverse=True)

        return {
            "date": target.isoformat(),
            "trade_count": len(day_trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate": len(wins) / len(day_trades),
            "by_strategy": {
                s: sum(t.pnl for t in day_trades if t.strategy == s)
                for s in set(t.strategy for t in day_trades)
            },
            "by_asset": {
                s: sum(t.pnl for t in day_trades if t.symbol == s)
                for s in set(t.symbol for t in day_trades)
            },
            "best_trade": sorted_trades[0].to_dict() if sorted_trades else None,
            "worst_trade": sorted_trades[-1].to_dict() if sorted_trades else None,
        }

    def save_report(self, report: PerformanceReport, filename: Optional[str] = None) -> Path:
        """Save report to disk."""
        filename = filename or f"report_{report.period_end.strftime('%Y%m%d')}.json"
        filepath = self.data_dir / filename

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved performance report to {filepath}")
        return filepath
