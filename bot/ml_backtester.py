"""
ML Model Backtester - Test ML trading strategies on historical data.

Features:
- Test ML models (random_forest, gradient_boosting, xgboost)
- Multiple symbols and timeframes
- Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)
- Walk-forward analysis
- Monte Carlo simulation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single backtest trade record."""

    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    side: str = "long"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    ml_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "quantity": self.quantity,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "exit_reason": self.exit_reason,
            "ml_confidence": round(self.ml_confidence, 4),
        }


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""

    # Basic
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L stats
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Time stats
    avg_hold_time_hours: float = 0.0
    avg_win_hold_time: float = 0.0
    avg_loss_hold_time: float = 0.0

    # Additional
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 2),
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "avg_trade": round(self.avg_trade, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "avg_drawdown": round(self.avg_drawdown, 2),
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "avg_hold_time_hours": round(self.avg_hold_time_hours, 1),
            "trades": [t.to_dict() for t in self.trades[-100:]],  # Last 100 trades
            "equity_curve": self.equity_curve[-500:],  # Last 500 points
        }


class MLBacktester:
    """
    ML Model Backtester - Test ML strategies on historical data.

    Usage:
        backtester = MLBacktester(
            model_type="gradient_boosting",
            initial_balance=10000,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )
        results = backtester.run("BTC/USDT", days=90)
        results.print_summary()
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        model_dir: str = "data/models",
        initial_balance: float = 10000.0,
        position_size_pct: float = 0.02,  # 2% of balance per trade
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit
        confidence_threshold: float = 0.6,
        commission_pct: float = 0.001,  # 0.1% per trade
    ):
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.confidence_threshold = confidence_threshold
        self.commission_pct = commission_pct

        self.balance = initial_balance
        self.position: Optional[BacktestTrade] = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []

        self._predictor = None

    def _load_model(self, symbol: str) -> bool:
        """Load ML model for symbol."""
        try:
            from bot.ml.predictor import MLPredictor

            symbol_base = symbol.replace("/", "_")
            model_name = f"{symbol_base}_{self.model_type}"

            self._predictor = MLPredictor(
                model_type=self.model_type,
                model_dir=str(self.model_dir),
            )

            if self._predictor.load(model_name):
                logger.info(f"Loaded model: {model_name}")
                return True

            # Try alternative model types
            for alt_type in ["random_forest", "gradient_boosting", "xgboost"]:
                if alt_type == self.model_type:
                    continue
                alt_name = f"{symbol_base}_{alt_type}"
                self._predictor = MLPredictor(
                    model_type=alt_type,
                    model_dir=str(self.model_dir),
                )
                if self._predictor.load(alt_name):
                    logger.info(f"Loaded alternative model: {alt_name}")
                    return True

            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _fetch_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch historical data from Yahoo Finance."""
        try:
            # Convert crypto symbol to Yahoo format
            yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=f"{days}d", interval="1h")

            if df.empty:
                logger.error(f"No data for {symbol}")
                return None

            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return None

    def run(
        self,
        symbol: str,
        days: int = 90,
        data: Optional[pd.DataFrame] = None,
    ) -> BacktestMetrics:
        """
        Run backtest for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            days: Number of days to backtest
            data: Optional pre-loaded OHLCV data

        Returns:
            BacktestMetrics with results
        """
        # Reset state
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []

        # Load model
        if not self._load_model(symbol):
            logger.warning(f"No model for {symbol}, using random signals")

        # Fetch data
        df = data if data is not None else self._fetch_data(symbol, days)
        if df is None or len(df) < 100:
            logger.error("Insufficient data for backtest")
            return BacktestMetrics(initial_balance=self.initial_balance)

        logger.info(f"Running backtest: {symbol}, {len(df)} bars, ${self.initial_balance}")

        # Run simulation
        for i in range(100, len(df)):  # Skip first 100 for indicators
            window = df.iloc[: i + 1]
            current = df.iloc[i]
            timestamp = df.index[i]

            # Check existing position
            if self.position:
                self._check_exit(current, timestamp, symbol)

            # Check for new entry
            if not self.position:
                self._check_entry(window, current, timestamp, symbol)

            # Record equity
            equity = self.balance
            if self.position:
                # Mark-to-market
                if self.position.side == "long":
                    equity += (
                        current["close"] - self.position.entry_price
                    ) * self.position.quantity
                else:
                    equity += (
                        self.position.entry_price - current["close"]
                    ) * self.position.quantity

            self.equity_curve.append(
                {
                    "timestamp": timestamp.isoformat()
                    if hasattr(timestamp, "isoformat")
                    else str(timestamp),
                    "equity": round(equity, 2),
                    "price": round(current["close"], 2),
                }
            )

        # Close any open position
        if self.position:
            last = df.iloc[-1]
            self._close_position(last["close"], df.index[-1], "End of backtest", symbol)

        return self._calculate_metrics()

    def _check_entry(
        self,
        window: pd.DataFrame,
        current: pd.Series,
        timestamp,
        symbol: str,
    ):
        """Check for entry signal."""
        if self._predictor and self._predictor.is_trained:
            try:
                prediction = self._predictor.predict(window)
                # PredictionResult has action and confidence attributes
                confidence = prediction.confidence
                action = prediction.action

                if confidence >= self.confidence_threshold:
                    if action == "LONG":
                        self._open_position("long", current, timestamp, symbol, confidence)
                    elif action == "SHORT":
                        self._open_position("short", current, timestamp, symbol, confidence)
            except Exception as e:
                logger.debug(f"Prediction error: {e}")
        else:
            # Simple momentum signal as fallback
            if len(window) >= 20:
                returns = window["close"].pct_change(10).iloc[-1]
                if returns > 0.02:
                    self._open_position("long", current, timestamp, symbol, 0.5)
                elif returns < -0.02:
                    self._open_position("short", current, timestamp, symbol, 0.5)

    def _open_position(
        self,
        side: str,
        current: pd.Series,
        timestamp,
        symbol: str,
        confidence: float,
    ):
        """Open a new position."""
        price = current["close"]
        position_value = self.balance * self.position_size_pct
        quantity = position_value / price

        # Apply commission
        commission = position_value * self.commission_pct
        self.balance -= commission

        self.position = BacktestTrade(
            entry_time=timestamp,
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            ml_confidence=confidence,
        )

        logger.debug(f"Opened {side} @ {price:.2f}, qty={quantity:.6f}")

    def _check_exit(self, current: pd.Series, timestamp, symbol: str):
        """Check for exit conditions."""
        if not self.position:
            return

        price = current["close"]
        high = current["high"]
        low = current["low"]

        exit_price = None
        exit_reason = ""

        if self.position.side == "long":
            # Stop loss
            sl_price = self.position.entry_price * (1 - self.stop_loss_pct)
            if low <= sl_price:
                exit_price = sl_price
                exit_reason = "Stop Loss"
            # Take profit
            tp_price = self.position.entry_price * (1 + self.take_profit_pct)
            if high >= tp_price:
                exit_price = tp_price
                exit_reason = "Take Profit"
        else:  # short
            # Stop loss
            sl_price = self.position.entry_price * (1 + self.stop_loss_pct)
            if high >= sl_price:
                exit_price = sl_price
                exit_reason = "Stop Loss"
            # Take profit
            tp_price = self.position.entry_price * (1 - self.take_profit_pct)
            if low <= tp_price:
                exit_price = tp_price
                exit_reason = "Take Profit"

        if exit_price:
            self._close_position(exit_price, timestamp, exit_reason, symbol)

    def _close_position(self, price: float, timestamp, reason: str, symbol: str):
        """Close current position."""
        if not self.position:
            return

        self.position.exit_time = timestamp
        self.position.exit_price = price
        self.position.exit_reason = reason

        # Calculate P&L
        if self.position.side == "long":
            self.position.pnl = (price - self.position.entry_price) * self.position.quantity
            self.position.pnl_pct = (price / self.position.entry_price - 1) * 100
        else:
            self.position.pnl = (self.position.entry_price - price) * self.position.quantity
            self.position.pnl_pct = (1 - price / self.position.entry_price) * 100

        # Apply commission
        commission = abs(self.position.pnl) * self.commission_pct
        self.position.pnl -= commission
        self.balance += self.position.pnl

        self.trades.append(self.position)
        logger.debug(f"Closed {self.position.side} @ {price:.2f}, PnL=${self.position.pnl:.2f}")

        self.position = None

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive metrics."""
        metrics = BacktestMetrics(
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

        if not self.trades:
            return metrics

        # Basic stats
        metrics.total_trades = len(self.trades)
        metrics.total_return = self.balance - self.initial_balance
        metrics.total_return_pct = (self.balance / self.initial_balance - 1) * 100

        # Win/loss
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(self.trades) if self.trades else 0

        # P&L stats
        win_pnls = [t.pnl for t in winners]
        loss_pnls = [abs(t.pnl) for t in losers]

        metrics.avg_win = np.mean(win_pnls) if win_pnls else 0
        metrics.avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        metrics.largest_win = max(win_pnls) if win_pnls else 0
        metrics.largest_loss = max(loss_pnls) if loss_pnls else 0
        metrics.avg_trade = np.mean([t.pnl for t in self.trades])

        # Profit factor
        total_wins = sum(win_pnls) if win_pnls else 0
        total_losses = sum(loss_pnls) if loss_pnls else 1
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Expectancy
        metrics.expectancy = (metrics.win_rate * metrics.avg_win) - (
            (1 - metrics.win_rate) * metrics.avg_loss
        )

        # Drawdown
        peak = self.initial_balance
        max_dd = 0
        dd_start = None
        max_dd_duration = 0
        drawdowns = []

        for point in self.equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity
                dd_start = None
            else:
                dd = peak - equity
                drawdowns.append(dd)
                if dd > max_dd:
                    max_dd = dd

        metrics.max_drawdown = max_dd
        metrics.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0
        metrics.avg_drawdown = np.mean(drawdowns) if drawdowns else 0

        # Calculate daily returns for Sharpe/Sortino
        if len(self.equity_curve) > 1:
            equities = [p["equity"] for p in self.equity_curve]
            daily_returns = pd.Series(equities).pct_change().dropna()
            metrics.daily_returns = daily_returns.tolist()

            if len(daily_returns) > 1 and daily_returns.std() > 0:
                # Sharpe (annualized, assuming hourly data -> 24*365 periods/year)
                excess_return = daily_returns.mean() - (0.02 / (24 * 365))
                metrics.sharpe_ratio = (excess_return / daily_returns.std()) * np.sqrt(24 * 365)

                # Sortino (only downside deviation)
                downside_returns = daily_returns[daily_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    metrics.sortino_ratio = (excess_return / downside_returns.std()) * np.sqrt(
                        24 * 365
                    )

        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            # Annualized return / max drawdown
            days = len(self.equity_curve) / 24  # Assuming hourly data
            annual_return = metrics.total_return_pct * (365 / days) if days > 0 else 0
            metrics.calmar_ratio = annual_return / metrics.max_drawdown_pct

        # Hold time
        hold_times = []
        win_hold_times = []
        loss_hold_times = []

        for t in self.trades:
            if t.entry_time and t.exit_time:
                try:
                    if hasattr(t.entry_time, "timestamp"):
                        duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                    else:
                        duration = 1  # Default 1 hour
                    hold_times.append(duration)
                    if t.pnl > 0:
                        win_hold_times.append(duration)
                    else:
                        loss_hold_times.append(duration)
                except (TypeError, AttributeError):
                    pass

        metrics.avg_hold_time_hours = np.mean(hold_times) if hold_times else 0
        metrics.avg_win_hold_time = np.mean(win_hold_times) if win_hold_times else 0
        metrics.avg_loss_hold_time = np.mean(loss_hold_times) if loss_hold_times else 0

        return metrics

    def print_summary(self, metrics: BacktestMetrics):
        """Print backtest summary."""
        print("\n" + "=" * 60)
        print("ML BACKTEST RESULTS")
        print("=" * 60)
        print(f"Model: {self.model_type}")
        print(f"Initial Balance: ${metrics.initial_balance:,.2f}")
        print(f"Final Balance: ${metrics.final_balance:,.2f}")
        print(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
        print()
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Winners: {metrics.winning_trades} | Losers: {metrics.losing_trades}")
        print(f"Win Rate: {metrics.win_rate:.1%}")
        print()
        print(f"Avg Win: ${metrics.avg_win:.2f} | Avg Loss: ${metrics.avg_loss:.2f}")
        print(
            f"Largest Win: ${metrics.largest_win:.2f} | Largest Loss: ${metrics.largest_loss:.2f}"
        )
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Expectancy: ${metrics.expectancy:.2f}")
        print()
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print()
        print(f"Avg Hold Time: {metrics.avg_hold_time_hours:.1f} hours")
        print("=" * 60)


def run_quick_backtest(
    symbol: str = "BTC/USDT",
    days: int = 90,
    model_type: str = "gradient_boosting",
    initial_balance: float = 10000.0,
) -> BacktestMetrics:
    """Quick backtest helper function."""
    backtester = MLBacktester(
        model_type=model_type,
        initial_balance=initial_balance,
    )
    metrics = backtester.run(symbol, days)
    backtester.print_summary(metrics)
    return metrics


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    logging.basicConfig(level=logging.INFO)
    run_quick_backtest(symbol, days)
