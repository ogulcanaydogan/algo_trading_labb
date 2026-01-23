"""
Strategy Performance Tracker.

Tracks and displays performance metrics for all trading strategies.
Provides a dashboard view of what's working and what's not.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Status of a strategy."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    TESTING = "testing"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""

    strategy_id: str
    strategy_name: str
    strategy_type: str  # e.g., "regime_sizing", "trend_following", "mean_reversion"

    # Status
    status: StrategyStatus = StrategyStatus.TESTING
    is_profitable: bool = False

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0

    # Time metrics
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    active_days: int = 0

    # Recent performance (last 30 days)
    recent_pnl: float = 0.0
    recent_trades: int = 0
    recent_win_rate: float = 0.0

    # Regime breakdown
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "status": self.status.value,
            "is_profitable": self.is_profitable,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "first_trade": self.first_trade.isoformat() if self.first_trade else None,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
            "active_days": self.active_days,
            "recent_pnl": self.recent_pnl,
            "recent_trades": self.recent_trades,
            "recent_win_rate": self.recent_win_rate,
            "regime_performance": self.regime_performance,
        }


@dataclass
class TradeEntry:
    """A single trade entry for tracking."""

    trade_id: str
    strategy_id: str
    symbol: str
    side: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    quantity: float

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    regime: Optional[str] = None
    exit_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "regime": self.regime,
            "exit_reason": self.exit_reason,
        }


class StrategyTracker:
    """
    Tracks performance of all strategies.

    Usage:
        tracker = StrategyTracker()

        # Register strategies
        tracker.register_strategy("regime_sizing_btc", "Regime Sizing BTC", "regime_sizing")

        # Record trades
        trade_id = tracker.record_entry("regime_sizing_btc", "BTC/USDT", "long", 50000.0, 0.1)
        tracker.record_exit(trade_id, 52000.0, regime="bull")

        # Get performance
        perf = tracker.get_performance("regime_sizing_btc")
        dashboard = tracker.get_dashboard()
    """

    def __init__(self, data_dir: Path = Path("data/strategy_tracker")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._strategies: Dict[str, StrategyPerformance] = {}
        self._trades: Dict[str, TradeEntry] = {}
        self._trade_history: List[TradeEntry] = []

        self._load_data()

    def register_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        strategy_type: str,
        status: StrategyStatus = StrategyStatus.TESTING,
    ) -> None:
        """Register a new strategy for tracking."""

        if strategy_id in self._strategies:
            logger.warning(f"Strategy {strategy_id} already registered")
            return

        self._strategies[strategy_id] = StrategyPerformance(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            status=status,
        )

        logger.info(f"Registered strategy: {strategy_name} ({strategy_id})")
        self._save_data()

    def update_strategy_status(
        self,
        strategy_id: str,
        status: StrategyStatus,
    ) -> None:
        """Update strategy status."""
        if strategy_id in self._strategies:
            self._strategies[strategy_id].status = status
            self._save_data()

    def record_entry(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        regime: Optional[str] = None,
    ) -> str:
        """Record a trade entry. Returns trade_id."""

        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy {strategy_id} not registered")

        trade_id = f"{strategy_id}_{datetime.now().timestamp()}"

        trade = TradeEntry(
            trade_id=trade_id,
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            entry_time=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            regime=regime,
        )

        self._trades[trade_id] = trade

        logger.info(f"Trade entry: {strategy_id} | {side} {quantity} {symbol} @ {entry_price}")

        return trade_id

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "signal",
        regime: Optional[str] = None,
    ) -> Optional[float]:
        """Record a trade exit. Returns P&L."""

        if trade_id not in self._trades:
            logger.warning(f"Trade {trade_id} not found")
            return None

        trade = self._trades[trade_id]
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        if regime:
            trade.regime = regime

        # Calculate P&L
        if trade.side == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)

        # Update strategy performance
        self._update_strategy_metrics(trade)

        # Move to history
        self._trade_history.append(trade)
        del self._trades[trade_id]

        logger.info(
            f"Trade exit: {trade.strategy_id} | {trade.side} @ {exit_price} | "
            f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct * 100:.2f}%)"
        )

        self._save_data()
        return trade.pnl

    def _update_strategy_metrics(self, trade: TradeEntry) -> None:
        """Update strategy performance metrics after a trade."""

        perf = self._strategies.get(trade.strategy_id)
        if not perf:
            return

        # Update trade counts
        perf.total_trades += 1
        if trade.pnl and trade.pnl > 0:
            perf.winning_trades += 1
        else:
            perf.losing_trades += 1

        # Update P&L
        if trade.pnl:
            perf.total_pnl += trade.pnl

            if trade.pnl > perf.best_trade:
                perf.best_trade = trade.pnl
            if trade.pnl < perf.worst_trade:
                perf.worst_trade = trade.pnl

        # Update time metrics
        if not perf.first_trade:
            perf.first_trade = trade.entry_time
        perf.last_trade = trade.exit_time

        # Calculate derived metrics
        perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0
        perf.is_profitable = perf.total_pnl > 0

        # Update averages
        strategy_trades = [t for t in self._trade_history if t.strategy_id == trade.strategy_id]
        wins = [t.pnl for t in strategy_trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in strategy_trades if t.pnl and t.pnl <= 0]

        perf.avg_win = sum(wins) / len(wins) if wins else 0
        perf.avg_loss = sum(losses) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        perf.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Regime performance
        if trade.regime:
            if trade.regime not in perf.regime_performance:
                perf.regime_performance[trade.regime] = {
                    "trades": 0,
                    "wins": 0,
                    "pnl": 0,
                }
            perf.regime_performance[trade.regime]["trades"] += 1
            if trade.pnl and trade.pnl > 0:
                perf.regime_performance[trade.regime]["wins"] += 1
            perf.regime_performance[trade.regime]["pnl"] += trade.pnl or 0

        # Recent performance (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_trades = [t for t in strategy_trades if t.exit_time and t.exit_time > recent_cutoff]
        perf.recent_trades = len(recent_trades)
        perf.recent_pnl = sum(t.pnl for t in recent_trades if t.pnl)
        recent_wins = len([t for t in recent_trades if t.pnl and t.pnl > 0])
        perf.recent_win_rate = recent_wins / len(recent_trades) if recent_trades else 0

    def get_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Get performance for a specific strategy."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> List[StrategyPerformance]:
        """Get all registered strategies."""
        return list(self._strategies.values())

    def get_dashboard(self) -> Dict:
        """
        Get dashboard data for all strategies.

        Returns a summary suitable for display.
        """

        strategies = list(self._strategies.values())

        # Sort by profitability
        profitable = [s for s in strategies if s.is_profitable]
        unprofitable = [s for s in strategies if not s.is_profitable]

        # Sort each group by total P&L
        profitable.sort(key=lambda s: s.total_pnl, reverse=True)
        unprofitable.sort(key=lambda s: s.total_pnl, reverse=True)

        # Calculate totals
        total_pnl = sum(s.total_pnl for s in strategies)
        total_trades = sum(s.total_trades for s in strategies)
        total_wins = sum(s.winning_trades for s in strategies)

        return {
            "summary": {
                "total_strategies": len(strategies),
                "profitable_strategies": len(profitable),
                "unprofitable_strategies": len(unprofitable),
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "overall_win_rate": total_wins / total_trades if total_trades > 0 else 0,
            },
            "top_performers": [s.to_dict() for s in profitable[:5]],
            "bottom_performers": [s.to_dict() for s in unprofitable[:5]],
            "all_strategies": [s.to_dict() for s in strategies],
            "by_type": self._group_by_type(strategies),
            "recent_activity": self._get_recent_activity(),
        }

    def _group_by_type(self, strategies: List[StrategyPerformance]) -> Dict:
        """Group strategies by type."""
        by_type: Dict[str, Dict] = {}

        for s in strategies:
            if s.strategy_type not in by_type:
                by_type[s.strategy_type] = {
                    "count": 0,
                    "total_pnl": 0,
                    "total_trades": 0,
                    "profitable_count": 0,
                }
            by_type[s.strategy_type]["count"] += 1
            by_type[s.strategy_type]["total_pnl"] += s.total_pnl
            by_type[s.strategy_type]["total_trades"] += s.total_trades
            if s.is_profitable:
                by_type[s.strategy_type]["profitable_count"] += 1

        return by_type

    def _get_recent_activity(self, days: int = 7) -> List[Dict]:
        """Get recent trading activity."""
        cutoff = datetime.now() - timedelta(days=days)

        recent = [t.to_dict() for t in self._trade_history if t.exit_time and t.exit_time > cutoff]

        return sorted(recent, key=lambda x: x["exit_time"], reverse=True)[:20]

    def get_strategy_comparison(self) -> str:
        """Get a formatted comparison of all strategies."""

        strategies = list(self._strategies.values())
        if not strategies:
            return "No strategies registered"

        # Sort by P&L
        strategies.sort(key=lambda s: s.total_pnl, reverse=True)

        lines = [
            "=" * 80,
            "STRATEGY PERFORMANCE DASHBOARD",
            "=" * 80,
            "",
            f"{'Strategy':<25} {'Status':<10} {'Trades':>8} {'Win%':>8} {'P&L':>12} {'PF':>8}",
            "-" * 80,
        ]

        for s in strategies:
            status_icon = {
                StrategyStatus.ACTIVE: "[ON]",
                StrategyStatus.PAUSED: "[||]",
                StrategyStatus.DISABLED: "[X]",
                StrategyStatus.TESTING: "[?]",
            }.get(s.status, "[?]")

            pnl_color = "+" if s.total_pnl > 0 else ""

            lines.append(
                f"{s.strategy_name:<25} {status_icon:<10} {s.total_trades:>8} "
                f"{s.win_rate * 100:>7.1f}% {pnl_color}{s.total_pnl:>11.2f} {s.profit_factor:>7.2f}"
            )

        lines.extend(
            [
                "-" * 80,
                "",
                "Legend: [ON]=Active [||]=Paused [X]=Disabled [?]=Testing",
                "PF = Profit Factor (>1 is profitable)",
            ]
        )

        return "\n".join(lines)

    def _save_data(self) -> None:
        """Save tracker data to disk."""
        try:
            # Save strategies
            strategies_file = self.data_dir / "strategies.json"
            with open(strategies_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._strategies.items()},
                    f,
                    indent=2,
                )

            # Save trade history
            history_file = self.data_dir / "trade_history.json"
            with open(history_file, "w") as f:
                json.dump(
                    [t.to_dict() for t in self._trade_history[-1000:]],  # Keep last 1000
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Failed to save tracker data: {e}")

    def _load_data(self) -> None:
        """Load tracker data from disk."""
        try:
            # Load strategies
            strategies_file = self.data_dir / "strategies.json"
            if strategies_file.exists():
                with open(strategies_file, "r") as f:
                    data = json.load(f)
                    for sid, sdata in data.items():
                        perf = StrategyPerformance(
                            strategy_id=sdata["strategy_id"],
                            strategy_name=sdata["strategy_name"],
                            strategy_type=sdata["strategy_type"],
                            status=StrategyStatus(sdata.get("status", "testing")),
                            total_trades=sdata.get("total_trades", 0),
                            winning_trades=sdata.get("winning_trades", 0),
                            losing_trades=sdata.get("losing_trades", 0),
                            total_pnl=sdata.get("total_pnl", 0),
                            is_profitable=sdata.get("is_profitable", False),
                            win_rate=sdata.get("win_rate", 0),
                            profit_factor=sdata.get("profit_factor", 0),
                        )
                        self._strategies[sid] = perf

            # Load trade history
            history_file = self.data_dir / "trade_history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    data = json.load(f)
                    for tdata in data:
                        trade = TradeEntry(
                            trade_id=tdata["trade_id"],
                            strategy_id=tdata["strategy_id"],
                            symbol=tdata["symbol"],
                            side=tdata["side"],
                            entry_time=datetime.fromisoformat(tdata["entry_time"]),
                            entry_price=tdata["entry_price"],
                            quantity=tdata["quantity"],
                            exit_time=datetime.fromisoformat(tdata["exit_time"])
                            if tdata.get("exit_time")
                            else None,
                            exit_price=tdata.get("exit_price"),
                            pnl=tdata.get("pnl"),
                            pnl_pct=tdata.get("pnl_pct"),
                            regime=tdata.get("regime"),
                            exit_reason=tdata.get("exit_reason"),
                        )
                        self._trade_history.append(trade)

            logger.info(
                f"Loaded tracker data: {len(self._strategies)} strategies, "
                f"{len(self._trade_history)} historical trades"
            )

        except Exception as e:
            logger.warning(f"Failed to load tracker data: {e}")


# Global tracker instance
_tracker: Optional[StrategyTracker] = None


def get_tracker() -> StrategyTracker:
    """Get the global strategy tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = StrategyTracker()
    return _tracker


# =============================================================================
# Convenient Strategy Registration Helpers
# =============================================================================


def register_new_strategy(
    name: str,
    strategy_type: str = "custom",
    symbol: str = "",
    description: str = "",
    active: bool = False,
) -> str:
    """
    Quick helper to register a new trading strategy.

    Args:
        name: Human-readable strategy name (e.g., "Momentum BTC")
        strategy_type: Type category (e.g., "regime_sizing", "trend_following", "mean_reversion")
        symbol: Trading symbol if applicable
        description: Optional description
        active: Whether to mark as active immediately

    Returns:
        strategy_id for tracking

    Example:
        # Register a new experimental strategy
        strategy_id = register_new_strategy(
            name="Momentum ETH",
            strategy_type="momentum",
            symbol="ETH/USDT",
            active=True
        )
    """
    tracker = get_tracker()

    # Generate ID from name
    strategy_id = name.lower().replace(" ", "_").replace("/", "_")

    # Register
    tracker.register_strategy(
        strategy_id=strategy_id,
        strategy_name=name,
        strategy_type=strategy_type,
        status=StrategyStatus.ACTIVE if active else StrategyStatus.TESTING,
    )

    logger.info(f"Registered new strategy: {name} ({strategy_id}) - Type: {strategy_type}")

    return strategy_id


def list_strategy_types() -> Dict[str, int]:
    """Get count of strategies by type."""
    tracker = get_tracker()
    strategies = tracker.get_all_strategies()

    by_type: Dict[str, int] = {}
    for s in strategies:
        by_type[s.strategy_type] = by_type.get(s.strategy_type, 0) + 1

    return by_type


def get_best_strategies(limit: int = 5) -> List[Dict]:
    """Get top performing strategies by P&L."""
    tracker = get_tracker()
    strategies = tracker.get_all_strategies()

    # Sort by P&L
    sorted_strategies = sorted(strategies, key=lambda s: s.total_pnl, reverse=True)

    return [s.to_dict() for s in sorted_strategies[:limit]]


def get_strategy_summary() -> str:
    """Get a quick text summary of all strategies."""
    tracker = get_tracker()
    dashboard = tracker.get_dashboard()

    summary = dashboard["summary"]

    lines = [
        "=== Strategy Summary ===",
        f"Total: {summary['total_strategies']} strategies",
        f"Profitable: {summary['profitable_strategies']}",
        f"Total P&L: ${summary['total_pnl']:,.2f}",
        f"Trades: {summary['total_trades']}",
        f"Win Rate: {summary['overall_win_rate'] * 100:.1f}%",
        "",
        "Top 3 Performers:",
    ]

    for s in dashboard["top_performers"][:3]:
        pnl = f"+${s['total_pnl']:.2f}" if s["total_pnl"] >= 0 else f"-${abs(s['total_pnl']):.2f}"
        lines.append(f"  - {s['strategy_name']}: {pnl}")

    return "\n".join(lines)
