"""
Regime-Aware Backtesting Engine.

Features:
- Walk-forward optimization with out-of-sample testing
- Realistic execution simulation (slippage, fees)
- Comprehensive performance metrics
- Risk engine integration
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_detector import MarketRegime, RegimeConfig, RegimeDetector, RegimeState
from .regime_risk_engine import (
    PortfolioState,
    RegimeRiskEngine,
    RiskCheckResult,
    RiskConfig,
    TradeRequest,
)
from .regime_strategies import (
    RegimeStrategySelector,
    SignalDirection,
    StrategyConfig,
    StrategySignal,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    # Capital
    initial_capital: float = 10000.0

    # Execution costs
    slippage_bps: float = 5.0  # 0.05%
    commission_bps: float = 10.0  # 0.10%

    # Walk-forward
    walk_forward_splits: int = 5
    train_pct: float = 0.70

    # Data
    timeframe: str = "1h"
    lookback_bars: int = 100

    # Simulation
    use_synthetic_stops: bool = True


@dataclass
class Trade:
    """Represents a completed trade."""

    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    regime: Optional[str] = None
    strategy: Optional[str] = None
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "fees": round(self.fees, 4),
            "regime": self.regime,
            "strategy": self.strategy,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    # Core metrics
    initial_capital: float = 10000.0
    final_equity: float = 10000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Exposure metrics
    avg_exposure: float = 0.0
    max_leverage_used: float = 0.0
    time_in_market_pct: float = 0.0

    # Turnover
    total_turnover: float = 0.0
    avg_holding_period_hours: float = 0.0

    # Cost analysis
    total_fees: float = 0.0
    total_slippage: float = 0.0

    # Regime breakdown
    regime_stats: Dict[str, Dict] = field(default_factory=dict)

    # Data
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

    # Period info
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "performance": {
                "initial_capital": self.initial_capital,
                "final_equity": round(self.final_equity, 2),
                "total_return": round(self.total_return, 2),
                "total_return_pct": round(self.total_return_pct * 100, 2),
                "cagr": round(self.cagr * 100, 2),
            },
            "risk": {
                "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
                "max_drawdown_duration_days": self.max_drawdown_duration_days,
                "sharpe_ratio": round(self.sharpe_ratio, 3),
                "sortino_ratio": round(self.sortino_ratio, 3),
                "calmar_ratio": round(self.calmar_ratio, 3),
            },
            "trades": {
                "total": self.total_trades,
                "winners": self.winning_trades,
                "losers": self.losing_trades,
                "win_rate": round(self.win_rate * 100, 1),
                "profit_factor": round(self.profit_factor, 2),
                "avg_win": round(self.avg_win, 2),
                "avg_loss": round(self.avg_loss, 2),
                "avg_trade": round(self.avg_trade, 2),
                "largest_win": round(self.largest_win, 2),
                "largest_loss": round(self.largest_loss, 2),
            },
            "exposure": {
                "avg_exposure_pct": round(self.avg_exposure * 100, 1),
                "max_leverage": round(self.max_leverage_used, 2),
                "time_in_market_pct": round(self.time_in_market_pct * 100, 1),
            },
            "costs": {
                "total_fees": round(self.total_fees, 2),
                "total_slippage": round(self.total_slippage, 2),
                "total_turnover": round(self.total_turnover, 2),
                "avg_holding_hours": round(self.avg_holding_period_hours, 1),
            },
            "regime_breakdown": self.regime_stats,
            "period": {
                "start": self.start_date.isoformat() if self.start_date else None,
                "end": self.end_date.isoformat() if self.end_date else None,
                "trading_days": self.trading_days,
            },
        }


class RegimeBacktester:
    """
    Walk-forward backtester with regime-aware strategy selection.

    Features:
    - Realistic execution simulation
    - Risk engine integration
    - Walk-forward optimization
    - Comprehensive reporting
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        strategy_config: Optional[StrategyConfig] = None,
    ):
        self.config = config or BacktestConfig()
        self.regime_config = regime_config or RegimeConfig()
        self.risk_config = risk_config or RiskConfig()
        self.strategy_config = strategy_config or StrategyConfig()

        # Components
        self.regime_detector = RegimeDetector(self.regime_config)
        self.risk_engine = RegimeRiskEngine(self.risk_config)
        self.strategy_selector = RegimeStrategySelector(self.strategy_config)

        # State
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._position: Optional[Trade] = None
        self._trades: List[Trade] = []
        self._equity_curve: List[Dict] = []
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: OHLCV DataFrame with datetime index
            symbol: Trading symbol

        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest: {len(df)} bars, symbol={symbol}")

        # Reset state
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._position = None
        self._trades = []
        self._equity_curve = []
        self._regime_history = []

        # Ensure proper column names
        df = df.copy()
        df.columns = df.columns.str.lower()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Initialize portfolio state for risk engine
        portfolio = PortfolioState(
            equity=self._equity,
            available_balance=self._equity,
            peak_equity=self._peak_equity,
        )
        self.risk_engine.update_portfolio(portfolio)

        # Main loop
        for i in range(self.config.lookback_bars, len(df)):
            bar_time = df.index[i]
            bar_data = df.iloc[: i + 1]
            current_bar = df.iloc[i]

            # Detect regime
            regime_state = self.regime_detector.detect(
                bar_data.tail(self.config.lookback_bars + 50),
                symbol=symbol,
                timeframe=self.config.timeframe,
            )
            self.risk_engine.update_regime(regime_state)
            self._regime_history.append((bar_time, regime_state.regime))

            # Update volatility for risk engine
            if regime_state.indicators:
                self.risk_engine.update_volatility(
                    regime_state.indicators.realized_vol,
                    regime_state.indicators.realized_vol * 0.8,  # Approximate normal
                )

            # Check if position should be closed (stop/take-profit)
            if self._position and self._position.is_open:
                self._check_exit(current_bar, bar_time)

            # Generate signal if no position
            if not self._position or not self._position.is_open:
                signal = self.strategy_selector.generate_signal(
                    bar_data.tail(self.config.lookback_bars + 50),
                    regime_state,
                )

                if signal and signal.is_entry:
                    self._try_enter(signal, current_bar, bar_time)

            # Record equity
            self._record_equity(bar_time, current_bar["close"], regime_state.regime)

        # Close any open position at end
        if self._position and self._position.is_open:
            self._close_position(df.iloc[-1]["close"], df.index[-1], "end_of_backtest")

        # Calculate results
        return self._calculate_results(df, symbol)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
    ) -> List[BacktestResult]:
        """
        Run walk-forward analysis with multiple train/test splits.

        Args:
            df: Full OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            List of BacktestResult for each out-of-sample period
        """
        logger.info(f"Starting walk-forward analysis: {self.config.walk_forward_splits} splits")

        results = []
        n_bars = len(df)
        split_size = n_bars // self.config.walk_forward_splits

        for i in range(self.config.walk_forward_splits):
            start_idx = i * split_size
            end_idx = min((i + 2) * split_size, n_bars)  # Overlap for continuity

            split_df = df.iloc[start_idx:end_idx]

            # Train/test split within this fold
            train_size = int(len(split_df) * self.config.train_pct)

            # Run on test portion only
            test_df = split_df.iloc[train_size:]

            if len(test_df) < self.config.lookback_bars + 50:
                continue

            logger.info(
                f"Walk-forward split {i + 1}/{self.config.walk_forward_splits}: "
                f"{len(test_df)} test bars"
            )

            result = self.run(test_df, symbol)
            results.append(result)

        return results

    def _try_enter(
        self,
        signal: StrategySignal,
        bar: pd.Series,
        bar_time: datetime,
    ) -> None:
        """Attempt to enter a position."""

        # Create trade request
        request = TradeRequest(
            symbol=signal.strategy_name,  # Use strategy name as symbol for tracking
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_confidence=signal.confidence,
            signal_reason=signal.reason,
        )

        # Check with risk engine
        result = self.risk_engine.check_trade(request)

        if not result.is_approved:
            logger.debug(f"Trade blocked: {result.block_reasons}")
            return

        # Apply slippage
        slippage = signal.entry_price * (self.config.slippage_bps / 10000)
        if signal.direction == SignalDirection.LONG:
            entry_price = signal.entry_price + slippage
        else:
            entry_price = signal.entry_price - slippage

        # Calculate position size
        quantity = result.approved_quantity

        # Commission
        commission = entry_price * quantity * (self.config.commission_bps / 10000)

        # Create trade
        self._position = Trade(
            symbol=request.symbol,
            direction=signal.direction.value,
            entry_time=bar_time,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            fees=commission,
            slippage=slippage * quantity,
            regime=signal.regime.value if signal.regime else None,
            strategy=signal.strategy_name,
        )

        # Deduct from equity
        self._equity -= commission

        logger.debug(
            f"Entered {signal.direction.value} @ {entry_price:.2f}, "
            f"qty={quantity:.4f}, stop={signal.stop_loss:.2f}"
        )

    def _check_exit(self, bar: pd.Series, bar_time: datetime) -> None:
        """Check if position should be closed."""

        if not self._position:
            return

        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        exit_price = None
        exit_reason = ""

        if self._position.direction == "long":
            # Check stop loss
            if low <= self._position.stop_loss:
                exit_price = self._position.stop_loss
                exit_reason = "stop_loss"
            # Check take profit
            elif self._position.take_profit and high >= self._position.take_profit:
                exit_price = self._position.take_profit
                exit_reason = "take_profit"

        else:  # short
            # Check stop loss
            if high >= self._position.stop_loss:
                exit_price = self._position.stop_loss
                exit_reason = "stop_loss"
            # Check take profit
            elif self._position.take_profit and low <= self._position.take_profit:
                exit_price = self._position.take_profit
                exit_reason = "take_profit"

        if exit_price:
            self._close_position(exit_price, bar_time, exit_reason)

    def _close_position(
        self,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ) -> None:
        """Close the current position."""

        if not self._position:
            return

        # Apply slippage
        slippage = exit_price * (self.config.slippage_bps / 10000)
        if self._position.direction == "long":
            exit_price = exit_price - slippage
        else:
            exit_price = exit_price + slippage

        # Commission
        commission = exit_price * self._position.quantity * (self.config.commission_bps / 10000)

        # Calculate P&L
        if self._position.direction == "long":
            pnl = (exit_price - self._position.entry_price) * self._position.quantity
        else:
            pnl = (self._position.entry_price - exit_price) * self._position.quantity

        pnl -= commission
        pnl_pct = pnl / (self._position.entry_price * self._position.quantity)

        # Update trade
        self._position.exit_time = exit_time
        self._position.exit_price = exit_price
        self._position.pnl = pnl
        self._position.pnl_pct = pnl_pct
        self._position.fees += commission
        self._position.slippage += slippage * self._position.quantity
        self._position.exit_reason = reason

        # Update equity
        self._equity += pnl + (self._position.entry_price * self._position.quantity)
        self._peak_equity = max(self._peak_equity, self._equity)

        # Record trade result for risk engine
        self.risk_engine.record_trade_result(pnl, pnl > 0)

        # Update portfolio state
        portfolio = PortfolioState(
            equity=self._equity,
            available_balance=self._equity,
            peak_equity=self._peak_equity,
        )
        self.risk_engine.update_portfolio(portfolio)

        # Store trade
        self._trades.append(self._position)
        self._position = None

        logger.debug(
            f"Closed {self._trades[-1].direction} @ {exit_price:.2f}, "
            f"pnl={pnl:.2f} ({pnl_pct:.2%}), reason={reason}"
        )

    def _record_equity(
        self,
        timestamp: datetime,
        price: float,
        regime: MarketRegime,
    ) -> None:
        """Record equity curve point."""

        # Calculate current equity including open position
        equity = self._equity
        if self._position and self._position.is_open:
            if self._position.direction == "long":
                unrealized = (price - self._position.entry_price) * self._position.quantity
            else:
                unrealized = (self._position.entry_price - price) * self._position.quantity
            equity += unrealized

        self._equity_curve.append(
            {
                "timestamp": timestamp.isoformat(),
                "equity": equity,
                "drawdown": (self._peak_equity - equity) / self._peak_equity
                if self._peak_equity > 0
                else 0,
                "regime": regime.value,
                "has_position": self._position is not None and self._position.is_open,
            }
        )

    def _calculate_results(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""

        result = BacktestResult(
            initial_capital=self.config.initial_capital,
            final_equity=self._equity,
            trades=self._trades,
            equity_curve=self._equity_curve,
        )

        # Period info
        result.start_date = df.index[0]
        result.end_date = df.index[-1]
        result.trading_days = (result.end_date - result.start_date).days

        # Returns
        result.total_return = self._equity - self.config.initial_capital
        result.total_return_pct = result.total_return / self.config.initial_capital

        # CAGR
        if result.trading_days > 0:
            years = result.trading_days / 365.25
            if years > 0 and result.final_equity > 0:
                result.cagr = (result.final_equity / self.config.initial_capital) ** (1 / years) - 1

        # Drawdown analysis
        if self._equity_curve:
            equity_series = pd.Series([e["equity"] for e in self._equity_curve])
            peak_series = equity_series.cummax()
            drawdown_series = (peak_series - equity_series) / peak_series

            result.max_drawdown_pct = drawdown_series.max()

            # Max drawdown duration
            in_drawdown = drawdown_series > 0
            drawdown_periods = (~in_drawdown).cumsum()
            if in_drawdown.any():
                dd_lengths = in_drawdown.groupby(drawdown_periods).sum()
                result.max_drawdown_duration_days = int(dd_lengths.max() / 24)  # Assuming hourly

        # Trade statistics
        result.total_trades = len(self._trades)
        if result.total_trades > 0:
            winners = [t for t in self._trades if t.is_winner]
            losers = [t for t in self._trades if not t.is_winner]

            result.winning_trades = len(winners)
            result.losing_trades = len(losers)
            result.win_rate = result.winning_trades / result.total_trades

            if winners:
                result.avg_win = sum(t.pnl for t in winners) / len(winners)
                result.largest_win = max(t.pnl for t in winners)
            if losers:
                result.avg_loss = sum(t.pnl for t in losers) / len(losers)
                result.largest_loss = min(t.pnl for t in losers)

            result.avg_trade = sum(t.pnl for t in self._trades) / result.total_trades

            # Profit factor
            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            # Costs
            result.total_fees = sum(t.fees for t in self._trades)
            result.total_slippage = sum(t.slippage for t in self._trades)

            # Turnover
            result.total_turnover = sum(
                t.entry_price * t.quantity * 2
                for t in self._trades  # Entry + exit
            )

            # Holding period
            holding_periods = []
            for t in self._trades:
                if t.exit_time and t.entry_time:
                    hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                    holding_periods.append(hours)
            if holding_periods:
                result.avg_holding_period_hours = sum(holding_periods) / len(holding_periods)

        # Risk-adjusted returns
        if self._equity_curve:
            equity_series = pd.Series([e["equity"] for e in self._equity_curve])
            returns = equity_series.pct_change().dropna()

            if len(returns) > 1 and returns.std() > 0:
                # Sharpe (assuming risk-free = 0)
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)

                # Sortino (downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        result.sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252 * 24)

        # Calmar ratio
        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.cagr / result.max_drawdown_pct

        # Exposure
        if self._equity_curve:
            positions_held = sum(1 for e in self._equity_curve if e.get("has_position"))
            result.time_in_market_pct = positions_held / len(self._equity_curve)

        # Regime breakdown
        regime_trades: Dict[str, List[Trade]] = {}
        for trade in self._trades:
            regime = trade.regime or "unknown"
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)

        for regime, trades in regime_trades.items():
            winners = [t for t in trades if t.is_winner]
            result.regime_stats[regime] = {
                "trades": len(trades),
                "win_rate": len(winners) / len(trades) if trades else 0,
                "total_pnl": sum(t.pnl for t in trades),
                "avg_pnl": sum(t.pnl for t in trades) / len(trades) if trades else 0,
            }

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"return={result.total_return_pct:.2%}, "
            f"sharpe={result.sharpe_ratio:.2f}, "
            f"max_dd={result.max_drawdown_pct:.2%}"
        )

        return result


def run_backtest_from_csv(
    csv_path: str,
    symbol: str = "BTC/USDT",
    output_path: Optional[str] = None,
) -> BacktestResult:
    """
    Convenience function to run backtest from CSV file.

    Args:
        csv_path: Path to OHLCV CSV file
        symbol: Trading symbol
        output_path: Optional path to save results JSON

    Returns:
        BacktestResult
    """
    logger.info(f"Loading data from {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    backtester = RegimeBacktester()
    result = backtester.run(df, symbol)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return result
