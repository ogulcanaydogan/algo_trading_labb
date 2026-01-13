"""
Backtesting and Strategy Testing Module

This module allows you to test your strategy on historical data and analyze its performance.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bot.strategy import compute_indicators, generate_signal, StrategyConfig, calculate_position_size


@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = "LONG"  # LONG or SHORT
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "exit_reason": self.exit_reason,
            "confidence": self.confidence,
        }


@dataclass
class BacktestResult:
    """Backtest results"""
    initial_balance: float
    final_balance: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "trades": [t.to_dict() for t in self.trades],
        }

    def print_summary(self):
        """Print results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Starting Balance: ${self.initial_balance:,.2f}")
        print(f"Ending Balance: ${self.final_balance:,.2f}")
        print(f"Total P&L: ${self.total_pnl:,.2f} ({self.total_pnl_pct:.2f}%)")
        print(f"\nTotal Trades: {self.total_trades}")
        print(f"Winners: {self.winning_trades} | Losers: {self.losing_trades}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print(f"Average Win: ${self.avg_win:.2f}")
        print(f"Average Loss: ${self.avg_loss:.2f}")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print(f"Max Drawdown: ${self.max_drawdown:.2f} ({self.max_drawdown_pct:.2f}%)")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print("="*60 + "\n")


class Backtester:
    """
    Backtest engine - tests strategy with historical data
    """

    def __init__(
        self,
        strategy_config: StrategyConfig,
        initial_balance: float = 10000.0,
    ):
        self.config = strategy_config
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def run(self, ohlcv: pd.DataFrame) -> BacktestResult:
        """
        Run backtest

        Args:
            ohlcv: OHLCV dataframe (columns: open, high, low, close, volume)

        Returns:
            BacktestResult: Test results
        """
        print(f"\nStarting backtest... ({len(ohlcv)} bars)")

        # Calculate indicators
        enriched = compute_indicators(ohlcv, self.config)

        # Test strategy for each bar
        for i in range(len(enriched)):
            if i < self.config.ema_slow + 5:
                continue

            current_bar = enriched.iloc[:i+1]
            current = current_bar.iloc[-1]

            # Check existing position
            if self.position:
                self._check_exit(current)

            # Check for new position
            if not self.position:
                signal = generate_signal(current_bar, self.config)
                if signal["decision"] != "FLAT":
                    self._open_position(current, signal)

            # Save equity curve
            self.equity_curve.append({
                "timestamp": current.name,
                "balance": self.balance,
                "price": float(current["close"]),
            })

        # Close open position if exists
        if self.position:
            last = enriched.iloc[-1]
            self._close_position(last["close"], last.name, "End of backtest")

        return self._calculate_results()

    def _open_position(self, bar: pd.Series, signal: Dict):
        """Open new position"""
        price = float(bar["close"])

        # Calculate position size
        size = calculate_position_size(
            self.balance,
            self.config.risk_per_trade_pct,
            price,
            self.config.stop_loss_pct,
        )

        # Set stop loss and take profit levels
        if signal["decision"] == "LONG":
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:  # SHORT
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)

        self.position = Trade(
            entry_time=bar.name,
            direction=signal["decision"],
            entry_price=price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=signal["confidence"],
        )

        print(f"{signal['decision']} position opened: ${price:.2f} | Size: {size:.4f}")

    def _check_exit(self, bar: pd.Series):
        """Check position exit"""
        if not self.position:
            return

        high = float(bar["high"])
        low = float(bar["low"])

        exit_price = None
        exit_reason = ""

        if self.position.direction == "LONG":
            # Stop loss check
            if low <= self.position.stop_loss:
                exit_price = self.position.stop_loss
                exit_reason = "Stop Loss"
            # Take profit check
            elif high >= self.position.take_profit:
                exit_price = self.position.take_profit
                exit_reason = "Take Profit"
        else:  # SHORT
            # Stop loss check
            if high >= self.position.stop_loss:
                exit_price = self.position.stop_loss
                exit_reason = "Stop Loss"
            # Take profit check
            elif low <= self.position.take_profit:
                exit_price = self.position.take_profit
                exit_reason = "Take Profit"

        if exit_price:
            self._close_position(exit_price, bar.name, exit_reason)

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Close position"""
        if not self.position:
            return

        self.position.exit_price = exit_price
        self.position.exit_time = exit_time
        self.position.exit_reason = reason

        # Calculate P&L
        if self.position.direction == "LONG":
            self.position.pnl = (exit_price - self.position.entry_price) * self.position.size
            self.position.pnl_pct = (exit_price / self.position.entry_price - 1) * 100
        else:  # SHORT
            self.position.pnl = (self.position.entry_price - exit_price) * self.position.size
            self.position.pnl_pct = (1 - exit_price / self.position.entry_price) * 100

        # Update balance
        self.balance += self.position.pnl

        # Save trade
        self.trades.append(self.position)

        print(
            "Position closed: "
            f"${exit_price:.2f} | P&L: ${self.position.pnl:.2f} "
            f"({self.position.pnl_pct:.2f}%) | {reason}"
        )

        self.position = None

    def _calculate_results(self) -> BacktestResult:
        """Calculate results"""
        result = BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

        if not self.trades:
            return result

        # Basic metrics
        result.total_trades = len(self.trades)
        result.winning_trades = len([t for t in self.trades if t.pnl > 0])
        result.losing_trades = len([t for t in self.trades if t.pnl <= 0])
        result.total_pnl = self.balance - self.initial_balance
        result.total_pnl_pct = (self.balance / self.initial_balance - 1) * 100

        # Win rate
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades

        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.trades if t.pnl <= 0]

        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1
        result.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0

        for point in self.equity_curve:
            balance = point["balance"]
            if balance > peak:
                peak = balance
            drawdown = peak - balance
            if drawdown > max_dd:
                max_dd = drawdown

        result.max_drawdown = max_dd
        result.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0

        # Sharpe ratio (annualized)
        # Convert trade returns to approximate daily returns for proper annualization
        returns = np.array([t.pnl_pct / 100 for t in self.trades])
        if len(returns) > 1:
            # Assume average of ~1 trade per day for annualization
            # Risk-free rate assumption: 2% annual = 0.02/252 daily
            risk_free_daily = 0.02 / 252
            excess_returns = returns - risk_free_daily
            if np.std(returns) > 0:
                # Annualized Sharpe = mean * sqrt(252) / std
                result.sharpe_ratio = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(252)
            else:
                result.sharpe_ratio = 0
        else:
            result.sharpe_ratio = 0

        return result


def save_backtest_results(result: BacktestResult, filename: str = "backtest_results.json"):
    """Save backtest results to JSON file"""
    with open(filename, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Results saved: {filename}")
