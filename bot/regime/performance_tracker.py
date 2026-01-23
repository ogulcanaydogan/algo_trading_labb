"""
Performance Tracker.

Features:
- Paper vs Backtest comparison
- Regime-specific performance analysis
- Daily/weekly/monthly summaries
- Telegram report generation
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .regime_detector import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record."""

    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float = 0.0
    pnl_pct: float = 0.0
    regime: str = ""
    source: str = ""  # "paper" or "backtest"
    strategy_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "regime": self.regime,
            "source": self.source,
            "strategy_id": self.strategy_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TradeRecord":
        return cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            side=data["side"],
            entry_price=data["entry_price"],
            exit_price=data.get("exit_price"),
            quantity=data["quantity"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]) if data.get("exit_time") else None,
            pnl=data.get("pnl", 0.0),
            pnl_pct=data.get("pnl_pct", 0.0),
            regime=data.get("regime", ""),
            source=data.get("source", ""),
            strategy_id=data.get("strategy_id", ""),
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for a period."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
        }


class PerformanceTracker:
    """
    Tracks and compares paper trading vs backtest performance.
    """

    def __init__(self, data_dir: str = "data/performance"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Trade records
        self._paper_trades: List[TradeRecord] = []
        self._backtest_trades: List[TradeRecord] = []

        # Equity curves
        self._paper_equity: List[Tuple[datetime, float]] = []
        self._backtest_equity: List[Tuple[datetime, float]] = []

        # Load saved data
        self._load_data()

    def _load_data(self):
        """Load saved trade data."""
        paper_file = self.data_dir / "paper_trades.json"
        backtest_file = self.data_dir / "backtest_trades.json"

        if paper_file.exists():
            try:
                with open(paper_file) as f:
                    data = json.load(f)
                    self._paper_trades = [TradeRecord.from_dict(t) for t in data]
            except Exception as e:
                logger.error(f"Failed to load paper trades: {e}")

        if backtest_file.exists():
            try:
                with open(backtest_file) as f:
                    data = json.load(f)
                    self._backtest_trades = [TradeRecord.from_dict(t) for t in data]
            except Exception as e:
                logger.error(f"Failed to load backtest trades: {e}")

    def _save_data(self):
        """Save trade data to disk."""
        paper_file = self.data_dir / "paper_trades.json"
        backtest_file = self.data_dir / "backtest_trades.json"

        try:
            with open(paper_file, "w") as f:
                json.dump([t.to_dict() for t in self._paper_trades], f, indent=2)
            with open(backtest_file, "w") as f:
                json.dump([t.to_dict() for t in self._backtest_trades], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trade data: {e}")

    def record_paper_trade(self, trade: TradeRecord):
        """Record a paper trade."""
        trade.source = "paper"
        self._paper_trades.append(trade)
        self._save_data()

    def record_backtest_trade(self, trade: TradeRecord):
        """Record a backtest trade."""
        trade.source = "backtest"
        self._backtest_trades.append(trade)
        self._save_data()

    def update_paper_equity(self, equity: float):
        """Update paper trading equity."""
        self._paper_equity.append((datetime.now(), equity))

    def update_backtest_equity(self, equity: float, timestamp: datetime):
        """Update backtest equity."""
        self._backtest_equity.append((timestamp, equity))

    def calculate_metrics(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Calculate performance metrics from trades."""
        if not trades:
            return PerformanceMetrics()

        completed = [t for t in trades if t.exit_price is not None]

        if not completed:
            return PerformanceMetrics(total_trades=len(trades))

        pnls = [t.pnl for t in completed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_win = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0

        metrics = PerformanceMetrics(
            total_trades=len(completed),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=sum(pnls),
            win_rate=len(wins) / len(completed) if completed else 0,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            profit_factor=total_win / total_loss if total_loss > 0 else float("inf"),
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
        )

        # Calculate Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns = np.array(pnls)
            metrics.sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            )

        return metrics

    def get_paper_metrics(
        self,
        days: int = None,
        strategy_id: str = None,
    ) -> PerformanceMetrics:
        """Get paper trading metrics."""
        trades = self._paper_trades

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [t for t in trades if t.entry_time >= cutoff]

        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]

        return self.calculate_metrics(trades)

    def get_backtest_metrics(
        self,
        days: int = None,
        strategy_id: str = None,
    ) -> PerformanceMetrics:
        """Get backtest metrics."""
        trades = self._backtest_trades

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [t for t in trades if t.entry_time >= cutoff]

        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]

        return self.calculate_metrics(trades)

    def compare_performance(self) -> Dict:
        """Compare paper vs backtest performance."""
        paper = self.get_paper_metrics()
        backtest = self.get_backtest_metrics()

        comparison = {
            "paper": paper.to_dict(),
            "backtest": backtest.to_dict(),
            "divergence": {},
        }

        # Calculate divergence
        if backtest.total_trades > 0 and paper.total_trades > 0:
            comparison["divergence"] = {
                "pnl_diff": paper.total_pnl - backtest.total_pnl,
                "pnl_diff_pct": (
                    (paper.total_pnl - backtest.total_pnl) / abs(backtest.total_pnl) * 100
                    if backtest.total_pnl != 0
                    else 0
                ),
                "win_rate_diff": paper.win_rate - backtest.win_rate,
                "trade_count_diff": paper.total_trades - backtest.total_trades,
            }

        return comparison

    def get_regime_performance(self, source: str = "paper") -> Dict[str, PerformanceMetrics]:
        """Get performance breakdown by regime."""
        trades = self._paper_trades if source == "paper" else self._backtest_trades

        regime_trades: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            regime = trade.regime or "unknown"
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)

        return {regime: self.calculate_metrics(trades) for regime, trades in regime_trades.items()}

    def get_daily_summary(self, date: datetime = None) -> Dict:
        """Get daily performance summary."""
        if date is None:
            date = datetime.now()

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        paper_trades = [t for t in self._paper_trades if start <= t.entry_time < end]

        return {
            "date": date.strftime("%Y-%m-%d"),
            "trades": len(paper_trades),
            "pnl": sum(t.pnl for t in paper_trades),
            "wins": len([t for t in paper_trades if t.pnl > 0]),
            "losses": len([t for t in paper_trades if t.pnl < 0]),
            "by_symbol": self._group_by_symbol(paper_trades),
            "by_regime": self._group_by_regime(paper_trades),
        }

    def _group_by_symbol(self, trades: List[TradeRecord]) -> Dict:
        """Group trades by symbol."""
        result = {}
        for trade in trades:
            if trade.symbol not in result:
                result[trade.symbol] = {"count": 0, "pnl": 0}
            result[trade.symbol]["count"] += 1
            result[trade.symbol]["pnl"] += trade.pnl
        return result

    def _group_by_regime(self, trades: List[TradeRecord]) -> Dict:
        """Group trades by regime."""
        result = {}
        for trade in trades:
            regime = trade.regime or "unknown"
            if regime not in result:
                result[regime] = {"count": 0, "pnl": 0}
            result[regime]["count"] += 1
            result[regime]["pnl"] += trade.pnl
        return result

    def generate_telegram_summary(self, period: str = "daily") -> str:
        """
        Generate a formatted Telegram summary message.

        Args:
            period: "daily", "weekly", or "monthly"

        Returns:
            Formatted message string
        """
        now = datetime.now()

        if period == "daily":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_name = "Daily"
        elif period == "weekly":
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            period_name = "Weekly"
        else:  # monthly
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            period_name = "Monthly"

        # Get trades for period
        paper_trades = [t for t in self._paper_trades if t.entry_time >= start]
        metrics = self.calculate_metrics(paper_trades)

        # Get regime breakdown
        regime_perf = {}
        for trade in paper_trades:
            regime = trade.regime or "unknown"
            if regime not in regime_perf:
                regime_perf[regime] = {"count": 0, "pnl": 0}
            regime_perf[regime]["count"] += 1
            regime_perf[regime]["pnl"] += trade.pnl

        # Format message
        pnl_emoji = "🟢" if metrics.total_pnl >= 0 else "🔴"

        msg = f"""
📊 *{period_name} Trading Summary*
{now.strftime("%Y-%m-%d %H:%M")}

{pnl_emoji} *P&L: ${metrics.total_pnl:,.2f}*

📈 *Statistics*
• Trades: {metrics.total_trades}
• Win Rate: {metrics.win_rate:.1%}
• Wins: {metrics.winning_trades} | Losses: {metrics.losing_trades}
• Best: ${metrics.best_trade:,.2f}
• Worst: ${metrics.worst_trade:,.2f}
• Profit Factor: {metrics.profit_factor:.2f}

🎯 *By Regime*"""

        for regime, data in sorted(regime_perf.items()):
            icon = self._get_regime_icon(regime)
            pnl_sign = "+" if data["pnl"] >= 0 else ""
            msg += f"\n• {icon} {regime}: {data['count']} trades, {pnl_sign}${data['pnl']:,.2f}"

        # Add comparison if we have backtest data
        comparison = self.compare_performance()
        if comparison.get("divergence"):
            div = comparison["divergence"]
            msg += f"""

📉 *Paper vs Backtest*
• P&L Divergence: {div.get("pnl_diff_pct", 0):+.1f}%
• Win Rate Diff: {div.get("win_rate_diff", 0):+.1%}"""

        return msg

    def _get_regime_icon(self, regime: str) -> str:
        """Get emoji icon for regime."""
        icons = {
            "bull": "🐂",
            "bear": "🐻",
            "crash": "💥",
            "sideways": "↔️",
            "high_vol": "🌊",
            "unknown": "❓",
        }
        return icons.get(regime.lower(), "❓")

    def get_status(self) -> Dict:
        """Get tracker status."""
        return {
            "paper_trades": len(self._paper_trades),
            "backtest_trades": len(self._backtest_trades),
            "paper_metrics": self.get_paper_metrics().to_dict(),
            "backtest_metrics": self.get_backtest_metrics().to_dict(),
            "comparison": self.compare_performance(),
            "regime_performance": {
                "paper": {k: v.to_dict() for k, v in self.get_regime_performance("paper").items()},
            },
        }


# Global tracker instance
_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get or create the global performance tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker
