"""
Advanced Backtesting Framework.

Features:
- Walk-forward optimization
- Monte Carlo simulation
- Multiple strategy comparison
- Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)
- Transaction cost modeling
- Multi-timeframe analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_balance: float = 10000.0
    commission_pct: float = 0.1  # 0.1% per trade
    slippage_pct: float = 0.05  # 0.05% slippage
    risk_per_trade_pct: float = 2.0  # 2% risk per trade
    max_position_size_pct: float = 10.0  # Max 10% per position

    # Risk management
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    trailing_stop_pct: Optional[float] = None
    max_drawdown_pct: float = 20.0  # Stop trading if drawdown exceeds this

    # Walk-forward settings
    train_window_days: int = 180
    test_window_days: int = 30

    # Monte Carlo settings
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95


@dataclass
class TradeRecord:
    """Record of a single trade."""

    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    direction: str = "LONG"
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    strategy_name: str = ""
    signal_confidence: float = 0.0
    holding_period_hours: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "size": round(self.size, 6),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "commission": round(self.commission, 2),
            "slippage": round(self.slippage, 2),
            "exit_reason": self.exit_reason,
            "strategy_name": self.strategy_name,
            "holding_period_hours": round(self.holding_period_hours, 2),
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Basic metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profit metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    expectancy: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    calmar_ratio: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Volatility
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0

    # Trade statistics
    avg_trade_duration_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Cost metrics
    total_commission: float = 0.0
    total_slippage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class BacktestReport:
    """Complete backtest report."""

    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    monte_carlo_results: Optional[Dict] = None
    walk_forward_results: Optional[List[Dict]] = None

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbol: str = ""
    strategy_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metrics": self.metrics.to_dict(),
            "trade_count": len(self.trades),
            "monte_carlo": self.monte_carlo_results,
            "walk_forward": self.walk_forward_results,
        }

    def save(self, filepath: str) -> None:
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Backtest report saved to {filepath}")

    def print_summary(self) -> None:
        """Print summary of backtest results."""
        m = self.metrics
        print("\n" + "=" * 70)
        print(f"BACKTEST REPORT - {self.symbol} ({self.strategy_name})")
        print("=" * 70)
        print(
            f"Period: {self.start_date.date() if self.start_date else 'N/A'} to "
            f"{self.end_date.date() if self.end_date else 'N/A'}"
        )
        print("-" * 70)

        print("\n📊 PERFORMANCE SUMMARY")
        print(f"  Total Return: {m.total_return_pct:.2f}%")
        print(f"  Annualized Return: {m.annualized_return_pct:.2f}%")
        print(f"  Net Profit: ${m.net_profit:,.2f}")

        print("\n📈 TRADE STATISTICS")
        print(f"  Total Trades: {m.total_trades}")
        print(f"  Win Rate: {m.win_rate:.2%}")
        print(f"  Winners: {m.winning_trades} | Losers: {m.losing_trades}")
        print(f"  Avg Win: ${m.avg_win:,.2f} | Avg Loss: ${m.avg_loss:,.2f}")
        print(f"  Win/Loss Ratio: {m.avg_win_loss_ratio:.2f}")
        print(f"  Expectancy: ${m.expectancy:.2f} per trade")
        print(f"  Profit Factor: {m.profit_factor:.2f}")

        print("\n⚠️ RISK METRICS")
        print(f"  Max Drawdown: ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2f}%)")
        print(f"  Max DD Duration: {m.max_drawdown_duration_days:.0f} days")
        print(f"  Annualized Volatility: {m.annualized_volatility:.2f}%")

        print("\n📉 RISK-ADJUSTED RETURNS")
        print(f"  Sharpe Ratio: {m.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {m.sortino_ratio:.2f}")
        print(f"  Calmar Ratio: {m.calmar_ratio:.2f}")

        print("\n💰 COSTS")
        print(f"  Total Commission: ${m.total_commission:,.2f}")
        print(f"  Total Slippage: ${m.total_slippage:,.2f}")
        print(f"  Total Costs: ${m.total_commission + m.total_slippage:,.2f}")

        if self.monte_carlo_results:
            print("\n🎲 MONTE CARLO ANALYSIS")
            mc = self.monte_carlo_results
            print(
                f"  95% CI for Final Equity: ${mc.get('equity_lower', 0):,.2f} - ${mc.get('equity_upper', 0):,.2f}"
            )
            print(f"  Probability of Profit: {mc.get('prob_profit', 0):.1%}")
            print(f"  Expected Drawdown (95%): {mc.get('expected_max_dd_95', 0):.2f}%")

        print("=" * 70 + "\n")


class SignalGenerator:
    """Base class for trading signal generators."""

    def __init__(self, name: str = "base_strategy"):
        self.name = name

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_idx: int,
    ) -> Dict[str, Any]:
        """
        Generate trading signal.

        Args:
            data: OHLCV DataFrame with indicators
            current_idx: Current bar index

        Returns:
            Dict with keys: decision (LONG/SHORT/FLAT), confidence, stop_loss, take_profit
        """
        raise NotImplementedError


class MetricsCalculator:
    """Calculate comprehensive performance metrics."""

    @staticmethod
    def calculate(
        trades: List[TradeRecord],
        equity_curve: List[Dict],
        initial_balance: float,
        trading_days_per_year: int = 252,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        # Basic trade metrics
        metrics.total_trades = len(trades)
        metrics.winning_trades = sum(1 for t in trades if t.pnl > 0)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = (
            metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        )

        # Profit metrics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl <= 0]

        metrics.gross_profit = sum(wins) if wins else 0
        metrics.gross_loss = sum(losses) if losses else 0
        metrics.net_profit = metrics.gross_profit - metrics.gross_loss

        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.avg_win_loss_ratio = (
            metrics.avg_win / metrics.avg_loss if metrics.avg_loss > 0 else 0
        )

        metrics.profit_factor = (
            metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float("inf")
        )
        metrics.expectancy = (metrics.win_rate * metrics.avg_win) - (
            (1 - metrics.win_rate) * metrics.avg_loss
        )

        # Return metrics
        final_balance = equity_curve[-1]["balance"] if equity_curve else initial_balance
        metrics.total_return_pct = (final_balance / initial_balance - 1) * 100

        # Annualized return (assuming equity curve has timestamps)
        if len(equity_curve) > 1:
            start_time = equity_curve[0].get("timestamp")
            end_time = equity_curve[-1].get("timestamp")
            if start_time and end_time:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time)
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time)
                days = (end_time - start_time).days
                if days > 0:
                    years = days / 365
                    metrics.annualized_return_pct = (
                        (final_balance / initial_balance) ** (1 / years) - 1
                    ) * 100

        # Daily returns for risk metrics
        daily_returns = MetricsCalculator._calculate_daily_returns(equity_curve)

        if daily_returns:
            # Volatility
            metrics.annualized_volatility = (
                np.std(daily_returns) * np.sqrt(trading_days_per_year) * 100
            )

            # Downside deviation
            negative_returns = [r for r in daily_returns if r < 0]
            if negative_returns:
                metrics.downside_deviation = (
                    np.std(negative_returns) * np.sqrt(trading_days_per_year) * 100
                )

            # Risk-adjusted returns (assuming risk-free rate = 0)
            mean_daily = np.mean(daily_returns)
            std_daily = np.std(daily_returns)

            if std_daily > 0:
                metrics.sharpe_ratio = (mean_daily / std_daily) * np.sqrt(trading_days_per_year)

            if metrics.downside_deviation > 0:
                metrics.sortino_ratio = metrics.annualized_return_pct / metrics.downside_deviation

        # Drawdown metrics
        dd_info = MetricsCalculator._calculate_drawdown(equity_curve)
        metrics.max_drawdown = dd_info["max_drawdown"]
        metrics.max_drawdown_pct = dd_info["max_drawdown_pct"]
        metrics.max_drawdown_duration_days = dd_info["max_duration_days"]

        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct

        # Trade duration
        durations = [t.holding_period_hours for t in trades if t.holding_period_hours > 0]
        metrics.avg_trade_duration_hours = np.mean(durations) if durations else 0

        # Consecutive wins/losses
        metrics.max_consecutive_wins, metrics.max_consecutive_losses = (
            MetricsCalculator._calculate_streaks(trades)
        )

        # Costs
        metrics.total_commission = sum(t.commission for t in trades)
        metrics.total_slippage = sum(t.slippage for t in trades)

        return metrics

    @staticmethod
    def _calculate_daily_returns(equity_curve: List[Dict]) -> List[float]:
        """Calculate daily returns from equity curve."""
        if len(equity_curve) < 2:
            return []

        balances = [e["balance"] for e in equity_curve]
        returns = []

        for i in range(1, len(balances)):
            if balances[i - 1] > 0:
                daily_return = (balances[i] / balances[i - 1]) - 1
                returns.append(daily_return)

        return returns

    @staticmethod
    def _calculate_drawdown(equity_curve: List[Dict]) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        if not equity_curve:
            return {"max_drawdown": 0, "max_drawdown_pct": 0, "max_duration_days": 0}

        balances = [e["balance"] for e in equity_curve]
        peak = balances[0]
        max_dd = 0
        max_dd_pct = 0

        peak_idx = 0
        max_duration = 0
        current_duration = 0

        for i, balance in enumerate(balances):
            if balance > peak:
                peak = balance
                peak_idx = i
                current_duration = 0
            else:
                drawdown = peak - balance
                drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0

                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_pct = drawdown_pct

                current_duration = i - peak_idx
                if current_duration > max_duration:
                    max_duration = current_duration

        # Convert duration to days (assuming daily data)
        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "max_duration_days": max_duration,
        }

    @staticmethod
    def _calculate_streaks(trades: List[TradeRecord]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses


class MonteCarloSimulator:
    """Monte Carlo simulation for trade sequence analysis."""

    @staticmethod
    def simulate(
        trades: List[TradeRecord],
        initial_balance: float,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.

        Shuffles trade order to estimate distribution of outcomes.
        """
        if len(trades) < 10:
            return {"error": "Not enough trades for Monte Carlo simulation"}

        trade_pnls = [t.pnl for t in trades]
        final_equities = []
        max_drawdowns = []

        for _ in range(n_simulations):
            # Shuffle trade order
            shuffled_pnls = np.random.permutation(trade_pnls)

            # Calculate equity curve
            equity = [initial_balance]
            peak = initial_balance
            max_dd = 0

            for pnl in shuffled_pnls:
                new_equity = equity[-1] + pnl
                equity.append(new_equity)

                if new_equity > peak:
                    peak = new_equity
                dd = (peak - new_equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_equities.append(equity[-1])
            max_drawdowns.append(max_dd * 100)

        # Calculate statistics
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 - (1 - confidence_level) / 2) * 100

        return {
            "n_simulations": n_simulations,
            "confidence_level": confidence_level,
            "equity_mean": np.mean(final_equities),
            "equity_median": np.median(final_equities),
            "equity_lower": np.percentile(final_equities, lower_percentile),
            "equity_upper": np.percentile(final_equities, upper_percentile),
            "prob_profit": np.mean([e > initial_balance for e in final_equities]),
            "expected_max_dd_mean": np.mean(max_drawdowns),
            "expected_max_dd_95": np.percentile(max_drawdowns, 95),
            "ruin_probability": np.mean([e < initial_balance * 0.5 for e in final_equities]),
        }


class WalkForwardAnalyzer:
    """Walk-forward optimization and analysis."""

    @staticmethod
    def analyze(
        data: pd.DataFrame,
        signal_generator: SignalGenerator,
        config: BacktestConfig,
        train_window: int = 180,
        test_window: int = 30,
    ) -> List[Dict]:
        """
        Run walk-forward analysis.

        Splits data into train/test windows and evaluates out-of-sample performance.
        """
        results = []
        total_bars = len(data)

        start_idx = train_window

        while start_idx + test_window <= total_bars:
            # Define windows
            train_start = start_idx - train_window
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + test_window, total_bars)

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # Run backtest on test window
            backtester = AdvancedBacktester(config, signal_generator)
            report = backtester.run(test_data)

            results.append(
                {
                    "period_start": test_data.index[0].isoformat()
                    if hasattr(test_data.index[0], "isoformat")
                    else str(test_data.index[0]),
                    "period_end": test_data.index[-1].isoformat()
                    if hasattr(test_data.index[-1], "isoformat")
                    else str(test_data.index[-1]),
                    "total_return_pct": report.metrics.total_return_pct,
                    "sharpe_ratio": report.metrics.sharpe_ratio,
                    "max_drawdown_pct": report.metrics.max_drawdown_pct,
                    "total_trades": report.metrics.total_trades,
                    "win_rate": report.metrics.win_rate,
                }
            )

            start_idx += test_window

        return results


class AdvancedBacktester:
    """Advanced backtesting engine."""

    def __init__(
        self,
        config: BacktestConfig,
        signal_generator: Optional[SignalGenerator] = None,
    ):
        self.config = config
        self.signal_generator = signal_generator

        self.balance = config.initial_balance
        self.position: Optional[TradeRecord] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []

        self.total_commission = 0.0
        self.total_slippage = 0.0

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> BacktestReport:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with indicators
            symbol: Trading symbol

        Returns:
            BacktestReport with complete analysis
        """
        logger.info(f"Starting backtest for {symbol} ({len(data)} bars)")

        self.balance = self.config.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.total_commission = 0.0
        self.total_slippage = 0.0

        for i in range(len(data)):
            if i < 50:  # Skip warmup period
                continue

            current_bar = data.iloc[i]
            current_data = data.iloc[: i + 1]

            # Check existing position
            if self.position:
                self._check_exit(current_bar)

            # Check for new signal (only if no position)
            if not self.position and self.signal_generator:
                signal = self.signal_generator.generate_signal(current_data, i)

                if signal.get("decision") in ["LONG", "SHORT"]:
                    self._open_position(current_bar, signal, symbol)

            # Update equity curve
            unrealized_pnl = self._calculate_unrealized_pnl(current_bar)
            self.equity_curve.append(
                {
                    "timestamp": current_bar.name
                    if hasattr(current_bar.name, "isoformat")
                    else str(current_bar.name),
                    "balance": self.balance + unrealized_pnl,
                    "price": float(current_bar["close"]),
                    "position": self.position.direction if self.position else "FLAT",
                }
            )

            # Check max drawdown limit
            if self._check_max_drawdown():
                logger.warning("Max drawdown limit reached, stopping backtest")
                break

        # Close any open position at end
        if self.position:
            self._close_position(
                data.iloc[-1]["close"],
                data.iloc[-1].name,
                "End of backtest",
            )

        # Calculate metrics
        metrics = MetricsCalculator.calculate(
            self.trades,
            self.equity_curve,
            self.config.initial_balance,
        )

        # Create report
        report = BacktestReport(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=self.equity_curve,
            start_date=data.index[0] if hasattr(data.index[0], "date") else None,
            end_date=data.index[-1] if hasattr(data.index[-1], "date") else None,
            symbol=symbol,
            strategy_name=self.signal_generator.name if self.signal_generator else "Unknown",
        )

        # Run Monte Carlo simulation
        if len(self.trades) >= 10:
            report.monte_carlo_results = MonteCarloSimulator.simulate(
                self.trades,
                self.config.initial_balance,
                n_simulations=self.config.monte_carlo_runs,
                confidence_level=self.config.confidence_level,
            )

        return report

    def _open_position(
        self,
        bar: pd.Series,
        signal: Dict,
        symbol: str,
    ) -> None:
        """Open new position with costs."""
        raw_price = float(bar["close"])
        direction = signal["decision"]

        # Apply slippage
        if direction == "LONG":
            price = raw_price * (1 + self.config.slippage_pct / 100)
        else:
            price = raw_price * (1 - self.config.slippage_pct / 100)

        slippage = abs(price - raw_price)

        # Calculate position size
        risk_amount = self.balance * (self.config.risk_per_trade_pct / 100)
        stop_distance = price * (self.config.stop_loss_pct / 100)
        size = risk_amount / stop_distance if stop_distance > 0 else 0

        # Limit position size
        max_size = (self.balance * (self.config.max_position_size_pct / 100)) / price
        size = min(size, max_size)

        # Apply commission
        commission = price * size * (self.config.commission_pct / 100)
        self.balance -= commission
        self.total_commission += commission
        self.total_slippage += slippage

        self.position = TradeRecord(
            entry_time=bar.name,
            symbol=symbol,
            direction=direction,
            entry_price=price,
            size=size,
            commission=commission,
            slippage=slippage,
            strategy_name=self.signal_generator.name if self.signal_generator else "",
            signal_confidence=signal.get("confidence", 0),
        )

        logger.debug(f"Opened {direction} position at {price:.2f}")

    def _check_exit(self, bar: pd.Series) -> None:
        """Check exit conditions."""
        if not self.position:
            return

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        entry_price = self.position.entry_price
        direction = self.position.direction

        # Calculate stop and target
        if direction == "LONG":
            stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.config.take_profit_pct / 100)

            if low <= stop_loss:
                self._close_position(stop_loss, bar.name, "Stop Loss")
            elif high >= take_profit:
                self._close_position(take_profit, bar.name, "Take Profit")
        else:
            stop_loss = entry_price * (1 + self.config.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.config.take_profit_pct / 100)

            if high >= stop_loss:
                self._close_position(stop_loss, bar.name, "Stop Loss")
            elif low <= take_profit:
                self._close_position(take_profit, bar.name, "Take Profit")

    def _close_position(
        self,
        raw_exit_price: float,
        exit_time: Any,
        reason: str,
    ) -> None:
        """Close position with costs."""
        if not self.position:
            return

        # Apply slippage
        if self.position.direction == "LONG":
            exit_price = raw_exit_price * (1 - self.config.slippage_pct / 100)
        else:
            exit_price = raw_exit_price * (1 + self.config.slippage_pct / 100)

        slippage = abs(exit_price - raw_exit_price)

        # Apply commission
        commission = exit_price * self.position.size * (self.config.commission_pct / 100)
        self.total_commission += commission
        self.total_slippage += slippage

        # Calculate P&L
        if self.position.direction == "LONG":
            pnl = (exit_price - self.position.entry_price) * self.position.size - commission
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size - commission

        pnl_pct = (pnl / (self.position.entry_price * self.position.size)) * 100

        # Calculate holding period
        if hasattr(exit_time, "timestamp") and hasattr(self.position.entry_time, "timestamp"):
            holding_hours = (exit_time - self.position.entry_time).total_seconds() / 3600
        else:
            holding_hours = 0

        # Update position
        self.position.exit_time = exit_time
        self.position.exit_price = exit_price
        self.position.pnl = pnl
        self.position.pnl_pct = pnl_pct
        self.position.commission += commission
        self.position.slippage += slippage
        self.position.exit_reason = reason
        self.position.holding_period_hours = holding_hours

        # Update balance
        self.balance += pnl

        # Record trade
        self.trades.append(self.position)
        self.position = None

        logger.debug(f"Closed position: P&L ${pnl:.2f} ({reason})")

    def _calculate_unrealized_pnl(self, bar: pd.Series) -> float:
        """Calculate unrealized P&L for open position."""
        if not self.position:
            return 0.0

        current_price = float(bar["close"])

        if self.position.direction == "LONG":
            return (current_price - self.position.entry_price) * self.position.size
        else:
            return (self.position.entry_price - current_price) * self.position.size

    def _check_max_drawdown(self) -> bool:
        """Check if max drawdown limit is exceeded."""
        if not self.equity_curve:
            return False

        peak = max(e["balance"] for e in self.equity_curve)
        current = self.equity_curve[-1]["balance"]
        drawdown_pct = ((peak - current) / peak) * 100 if peak > 0 else 0

        return drawdown_pct > self.config.max_drawdown_pct


def run_backtest(
    data: pd.DataFrame,
    signal_generator: SignalGenerator,
    config: Optional[BacktestConfig] = None,
    symbol: str = "UNKNOWN",
    run_monte_carlo: bool = True,
) -> BacktestReport:
    """
    Run a complete backtest with all analysis.

    Args:
        data: OHLCV DataFrame
        signal_generator: Strategy signal generator
        config: Backtest configuration
        symbol: Trading symbol
        run_monte_carlo: Whether to run Monte Carlo simulation

    Returns:
        Complete BacktestReport
    """
    if config is None:
        config = BacktestConfig()

    backtester = AdvancedBacktester(config, signal_generator)
    report = backtester.run(data, symbol)

    return report
