"""
A/B Testing Framework for Strategy Comparison

This module provides a comprehensive framework for:
- Running multiple strategies in parallel (paper mode)
- Tracking performance with isolated paper portfolios
- Unified data feed for fair comparison
- Statistical analysis (t-tests, Sharpe ratio comparison, confidence intervals)
- JSON dashboard output

Usage:
    from bot.testing.ab_framework import ABTest, ABTestConfig
    from bot.strategy_interface import EMACrossoverStrategy, MomentumStrategy
    
    config = ABTestConfig(
        initial_balance=10000,
        test_duration_bars=1000,
        symbols=["BTC/USDT"]
    )
    
    ab_test = ABTest(config)
    ab_test.register_strategy("ema", EMACrossoverStrategy())
    ab_test.register_strategy("momentum", MomentumStrategy())
    
    result = ab_test.run(market_data)
    result.save_dashboard("ab_test_results.json")
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

# Import strategy interface
try:
    from bot.strategy_interface import (
        Strategy,
        StrategySignal,
        MarketState,
        StrategyAction,
        StrategyPerformance,
    )
except ImportError:
    # Fallback for standalone testing
    from dataclasses import dataclass as fallback_dataclass
    Strategy = object
    StrategySignal = object
    MarketState = object
    StrategyAction = object
    StrategyPerformance = object

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ABTestConfig:
    """Configuration for A/B test"""
    
    # Portfolio settings
    initial_balance: float = 10000.0
    trading_fee_pct: float = 0.1  # 0.1% per trade
    slippage_pct: float = 0.05  # 0.05% slippage
    
    # Test settings
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    test_duration_bars: Optional[int] = None  # None = use all data
    warmup_bars: int = 50  # Bars to skip for indicator warmup
    
    # Risk settings
    max_position_pct: float = 100.0  # Max % of portfolio per position
    max_drawdown_stop: float = 25.0  # Stop test if drawdown exceeds this %
    
    # Statistical settings
    confidence_level: float = 0.95  # For confidence intervals
    min_trades_for_significance: int = 30  # Min trades for statistical tests
    
    # Output settings
    output_dir: str = "ab_test_results"
    save_trade_history: bool = True
    save_equity_curve: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_balance": self.initial_balance,
            "trading_fee_pct": self.trading_fee_pct,
            "slippage_pct": self.slippage_pct,
            "symbols": self.symbols,
            "test_duration_bars": self.test_duration_bars,
            "warmup_bars": self.warmup_bars,
            "max_position_pct": self.max_position_pct,
            "max_drawdown_stop": self.max_drawdown_stop,
            "confidence_level": self.confidence_level,
            "min_trades_for_significance": self.min_trades_for_significance,
        }


# ============================================================
# Paper Portfolio
# ============================================================

@dataclass 
class Trade:
    """Record of a single trade"""
    trade_id: str
    strategy_name: str
    symbol: str
    direction: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    fees: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "size": self.size,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 6),
            "exit_reason": self.exit_reason,
            "fees": round(self.fees, 4),
        }


class PaperPortfolio:
    """
    Simulated portfolio for paper trading.
    
    Each strategy gets its own isolated portfolio to ensure
    fair comparison without position conflicts.
    """
    
    def __init__(
        self,
        strategy_name: str,
        initial_balance: float,
        trading_fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ):
        self.strategy_name = strategy_name
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.trading_fee_pct = trading_fee_pct
        self.slippage_pct = slippage_pct
        
        # Positions and trades
        self.positions: Dict[str, Trade] = {}  # symbol -> open trade
        self.closed_trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.peak_equity: float = initial_balance
        self.max_drawdown: float = 0.0
        self.max_drawdown_pct: float = 0.0
        
        # Statistics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.gross_profit: float = 0.0
        self.gross_loss: float = 0.0
        self.total_fees: float = 0.0
        
        logger.debug(f"Paper portfolio created for {strategy_name}: ${initial_balance:,.2f}")
    
    @property
    def equity(self) -> float:
        """Current total equity (cash + open position value)"""
        return self.cash + sum(
            self._calculate_position_value(trade)
            for trade in self.positions.values()
        )
    
    @property
    def pnl(self) -> float:
        """Total P&L"""
        return self.equity - self.initial_balance
    
    @property
    def pnl_pct(self) -> float:
        """Total P&L percentage"""
        if self.initial_balance == 0:
            return 0.0
        return (self.equity / self.initial_balance - 1) * 100
    
    @property
    def win_rate(self) -> float:
        """Win rate of closed trades"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss"""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss
    
    @property
    def avg_win(self) -> float:
        """Average winning trade"""
        if self.winning_trades == 0:
            return 0.0
        return self.gross_profit / self.winning_trades
    
    @property
    def avg_loss(self) -> float:
        """Average losing trade"""
        if self.losing_trades == 0:
            return 0.0
        return self.gross_loss / self.losing_trades
    
    def _calculate_position_value(self, trade: Trade) -> float:
        """Calculate current value of an open position"""
        # For simplicity, return 0 if no current price is tracked
        # This will be updated during the update_prices call
        return getattr(trade, '_current_value', 0.0)
    
    def _apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """Apply slippage to price"""
        slippage_factor = self.slippage_pct / 100
        
        if direction == "long":
            if is_entry:
                return price * (1 + slippage_factor)  # Pay more on entry
            else:
                return price * (1 - slippage_factor)  # Get less on exit
        else:  # short
            if is_entry:
                return price * (1 - slippage_factor)  # Get less on entry (selling)
            else:
                return price * (1 + slippage_factor)  # Pay more on exit (buying back)
    
    def _calculate_fee(self, notional: float) -> float:
        """Calculate trading fee"""
        return notional * (self.trading_fee_pct / 100)
    
    def open_position(
        self,
        symbol: str,
        direction: str,
        price: float,
        size_pct: float,
        timestamp: datetime,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
    ) -> Optional[Trade]:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            price: Entry price
            size_pct: Position size as % of equity
            timestamp: Entry timestamp
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            
        Returns:
            Trade object if successful, None otherwise
        """
        # Don't open if already have position in this symbol
        if symbol in self.positions:
            logger.debug(f"{self.strategy_name}: Already have position in {symbol}")
            return None
        
        # Calculate position size
        position_value = self.equity * (size_pct / 100)
        if position_value > self.cash:
            position_value = self.cash
        
        if position_value <= 0:
            logger.debug(f"{self.strategy_name}: Insufficient funds for position")
            return None
        
        # Apply slippage
        exec_price = self._apply_slippage(price, direction, is_entry=True)
        
        # Calculate size in units
        size = position_value / exec_price
        
        # Calculate and deduct fee
        fee = self._calculate_fee(position_value)
        self.total_fees += fee
        
        # Deduct from cash
        self.cash -= position_value + fee
        
        # Create trade
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            strategy_name=self.strategy_name,
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=exec_price,
            size=size,
            fees=fee,
        )
        
        # Store stop/take profit
        trade._stop_loss_pct = stop_loss_pct
        trade._take_profit_pct = take_profit_pct
        trade._current_value = size * exec_price  # Initial value
        
        self.positions[symbol] = trade
        
        logger.debug(
            f"{self.strategy_name}: Opened {direction} {symbol} @ {exec_price:.2f}, "
            f"size={size:.6f}, value=${position_value:.2f}"
        )
        
        return trade
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str = "signal",
    ) -> Optional[Trade]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
            
        Returns:
            Closed Trade object if successful, None otherwise
        """
        if symbol not in self.positions:
            return None
        
        trade = self.positions.pop(symbol)
        
        # Apply slippage
        exec_price = self._apply_slippage(price, trade.direction, is_entry=False)
        
        # Calculate P&L
        if trade.direction == "long":
            pnl = (exec_price - trade.entry_price) * trade.size
        else:  # short
            pnl = (trade.entry_price - exec_price) * trade.size
        
        pnl_pct = (pnl / (trade.entry_price * trade.size)) * 100
        
        # Exit fee
        exit_value = exec_price * trade.size
        exit_fee = self._calculate_fee(exit_value)
        self.total_fees += exit_fee
        
        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = exec_price
        trade.pnl = pnl - exit_fee
        trade.pnl_pct = pnl_pct
        trade.exit_reason = reason
        trade.fees += exit_fee
        
        # Return cash
        self.cash += exit_value - exit_fee
        
        # Update statistics
        self.closed_trades.append(trade)
        self.total_trades += 1
        
        if trade.pnl > 0:
            self.winning_trades += 1
            self.gross_profit += trade.pnl
        else:
            self.losing_trades += 1
            self.gross_loss += abs(trade.pnl)
        
        logger.debug(
            f"{self.strategy_name}: Closed {trade.direction} {symbol} @ {exec_price:.2f}, "
            f"PnL=${trade.pnl:.2f} ({trade.pnl_pct:.2f}%), reason={reason}"
        )
        
        return trade
    
    def update_prices(
        self,
        prices: Dict[str, float],
        timestamp: datetime,
    ):
        """
        Update current prices and check stops.
        
        Args:
            prices: Dict of symbol -> current price
            timestamp: Current timestamp
        """
        for symbol, trade in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            trade._current_value = trade.size * price
            
            # Check stop loss and take profit
            if trade.direction == "long":
                pnl_pct = (price / trade.entry_price - 1) * 100
            else:
                pnl_pct = (1 - price / trade.entry_price) * 100
            
            stop_loss_pct = getattr(trade, '_stop_loss_pct', 2.0)
            take_profit_pct = getattr(trade, '_take_profit_pct', 4.0)
            
            if pnl_pct <= -stop_loss_pct:
                self.close_position(symbol, price, timestamp, reason="stop_loss")
            elif pnl_pct >= take_profit_pct:
                self.close_position(symbol, price, timestamp, reason="take_profit")
        
        # Update equity curve and drawdown
        current_equity = self.equity
        self.equity_curve.append({
            "timestamp": timestamp.isoformat(),
            "equity": current_equity,
            "cash": self.cash,
            "positions_value": current_equity - self.cash,
        })
        
        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = self.peak_equity - current_equity
        drawdown_pct = (drawdown / self.peak_equity) * 100 if self.peak_equity > 0 else 0
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: float = 252) -> float:
        """Calculate Sharpe ratio from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        equities = [e["equity"] for e in self.equity_curve]
        returns = np.diff(equities) / equities[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualize
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0, periods_per_year: float = 252) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equities = [e["equity"] for e in self.equity_curve]
        returns = np.diff(equities) / equities[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        # Downside returns only
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        excess_return = np.mean(returns) - (risk_free_rate / periods_per_year)
        sortino = excess_return / downside_std * np.sqrt(periods_per_year)
        
        return float(sortino)
    
    def get_trade_returns(self) -> List[float]:
        """Get list of trade returns for statistical analysis"""
        return [t.pnl_pct for t in self.closed_trades]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            "strategy_name": self.strategy_name,
            "initial_balance": self.initial_balance,
            "final_equity": round(self.equity, 2),
            "total_pnl": round(self.pnl, 2),
            "total_pnl_pct": round(self.pnl_pct, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4) if self.profit_factor != float('inf') else "inf",
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.calculate_sharpe_ratio(), 4),
            "sortino_ratio": round(self.calculate_sortino_ratio(), 4),
            "total_fees": round(self.total_fees, 2),
        }


# ============================================================
# Strategy Runner
# ============================================================

class StrategyRunner:
    """
    Runs a strategy with its own paper portfolio.
    
    Handles signal generation and trade execution.
    """
    
    def __init__(
        self,
        name: str,
        strategy: Strategy,
        portfolio: PaperPortfolio,
    ):
        self.name = name
        self.strategy = strategy
        self.portfolio = portfolio
        self.signals_generated: int = 0
        self.signals_executed: int = 0
        self.last_signal: Optional[StrategySignal] = None
    
    def process_bar(
        self,
        state: MarketState,
        prices: Dict[str, float],
        timestamp: datetime,
    ):
        """
        Process a single bar of data.
        
        1. Update prices and check stops
        2. Generate signal from strategy
        3. Execute signal if actionable
        """
        # Update prices first (may close positions on stops)
        self.portfolio.update_prices(prices, timestamp)
        
        # Generate signal
        try:
            signal = self.strategy.predict(state)
            self.signals_generated += 1
            self.last_signal = signal
        except Exception as e:
            logger.error(f"Strategy {self.name} failed to predict: {e}")
            return
        
        # Execute signal
        symbol = state.symbol
        current_price = prices.get(symbol, state.close)
        
        # Determine action
        if signal.action in [StrategyAction.BUY, StrategyAction.LONG]:
            # Close any short position first
            if symbol in self.portfolio.positions:
                if self.portfolio.positions[symbol].direction == "short":
                    self.portfolio.close_position(symbol, current_price, timestamp, "reverse")
                else:
                    return  # Already long
            
            # Open long
            if signal.confidence >= 0.4:  # Min confidence threshold
                trade = self.portfolio.open_position(
                    symbol=symbol,
                    direction="long",
                    price=current_price,
                    size_pct=signal.position_size_pct,
                    timestamp=timestamp,
                    stop_loss_pct=signal.stop_loss_pct,
                    take_profit_pct=signal.take_profit_pct,
                )
                if trade:
                    self.signals_executed += 1
        
        elif signal.action in [StrategyAction.SELL, StrategyAction.SHORT]:
            # Close any long position first
            if symbol in self.portfolio.positions:
                if self.portfolio.positions[symbol].direction == "long":
                    self.portfolio.close_position(symbol, current_price, timestamp, "reverse")
                else:
                    return  # Already short
            
            # Open short
            if signal.confidence >= 0.4:
                trade = self.portfolio.open_position(
                    symbol=symbol,
                    direction="short",
                    price=current_price,
                    size_pct=signal.position_size_pct,
                    timestamp=timestamp,
                    stop_loss_pct=signal.stop_loss_pct,
                    take_profit_pct=signal.take_profit_pct,
                )
                if trade:
                    self.signals_executed += 1
        
        elif signal.action == StrategyAction.CLOSE:
            if symbol in self.portfolio.positions:
                self.portfolio.close_position(symbol, current_price, timestamp, "signal")
                self.signals_executed += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get runner summary"""
        portfolio_summary = self.portfolio.get_summary()
        portfolio_summary.update({
            "signals_generated": self.signals_generated,
            "signals_executed": self.signals_executed,
            "execution_rate": (
                round(self.signals_executed / self.signals_generated, 4)
                if self.signals_generated > 0 else 0
            ),
        })
        return portfolio_summary


# ============================================================
# Statistical Comparison
# ============================================================

@dataclass
class StatisticalComparison:
    """
    Statistical comparison between two strategies.
    
    Includes:
    - T-test on returns
    - Sharpe ratio comparison
    - Confidence intervals
    - Effect size (Cohen's d)
    """
    
    strategy_a: str
    strategy_b: str
    
    # T-test results
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    
    # Returns comparison
    mean_return_a: float = 0.0
    mean_return_b: float = 0.0
    std_return_a: float = 0.0
    std_return_b: float = 0.0
    
    # Sharpe comparison
    sharpe_a: float = 0.0
    sharpe_b: float = 0.0
    sharpe_diff: float = 0.0
    sharpe_winner: str = ""
    
    # Confidence interval on difference
    mean_diff: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    # Effect size
    cohens_d: float = 0.0
    effect_size_interpretation: str = ""
    
    # Sample sizes
    n_trades_a: int = 0
    n_trades_b: int = 0
    
    # Winner
    winner: str = ""
    confidence: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "t_test": {
                "t_statistic": round(float(self.t_statistic), 4),
                "p_value": round(float(self.p_value), 6),
                "is_significant": bool(self.is_significant),
            },
            "returns": {
                "mean_a": round(float(self.mean_return_a), 4),
                "mean_b": round(float(self.mean_return_b), 4),
                "std_a": round(float(self.std_return_a), 4),
                "std_b": round(float(self.std_return_b), 4),
                "mean_diff": round(float(self.mean_diff), 4),
                "ci_95": [round(float(self.ci_lower), 4), round(float(self.ci_upper), 4)],
            },
            "sharpe": {
                "sharpe_a": round(float(self.sharpe_a), 4),
                "sharpe_b": round(float(self.sharpe_b), 4),
                "sharpe_diff": round(float(self.sharpe_diff), 4),
                "winner": self.sharpe_winner,
            },
            "effect_size": {
                "cohens_d": round(float(self.cohens_d), 4),
                "interpretation": self.effect_size_interpretation,
            },
            "sample_sizes": {
                "n_a": self.n_trades_a,
                "n_b": self.n_trades_b,
            },
            "conclusion": {
                "winner": self.winner,
                "confidence": self.confidence,
            },
        }
    
    @staticmethod
    def compare(
        name_a: str,
        returns_a: List[float],
        sharpe_a: float,
        name_b: str,
        returns_b: List[float],
        sharpe_b: float,
        confidence_level: float = 0.95,
    ) -> "StatisticalComparison":
        """
        Perform statistical comparison between two strategies.
        
        Args:
            name_a: Name of strategy A
            returns_a: List of trade returns (%) for A
            sharpe_a: Sharpe ratio for A
            name_b: Name of strategy B
            returns_b: List of trade returns (%) for B
            sharpe_b: Sharpe ratio for B
            confidence_level: Confidence level for CI (default 0.95)
            
        Returns:
            StatisticalComparison object with all metrics
        """
        comparison = StatisticalComparison(
            strategy_a=name_a,
            strategy_b=name_b,
            n_trades_a=len(returns_a),
            n_trades_b=len(returns_b),
        )
        
        # Need sufficient data for meaningful statistics
        if len(returns_a) < 2 or len(returns_b) < 2:
            comparison.confidence = "insufficient_data"
            return comparison
        
        arr_a = np.array(returns_a)
        arr_b = np.array(returns_b)
        
        # Basic stats
        comparison.mean_return_a = float(np.mean(arr_a))
        comparison.mean_return_b = float(np.mean(arr_b))
        comparison.std_return_a = float(np.std(arr_a, ddof=1))
        comparison.std_return_b = float(np.std(arr_b, ddof=1))
        
        # Sharpe comparison
        comparison.sharpe_a = sharpe_a
        comparison.sharpe_b = sharpe_b
        comparison.sharpe_diff = sharpe_a - sharpe_b
        comparison.sharpe_winner = name_a if sharpe_a > sharpe_b else name_b
        
        # T-test (Welch's t-test for unequal variances)
        try:
            t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
            comparison.t_statistic = float(t_stat)
            comparison.p_value = float(p_value)
            comparison.is_significant = p_value < (1 - confidence_level)
        except Exception as e:
            logger.warning(f"T-test failed: {e}")
            comparison.p_value = 1.0
        
        # Mean difference and confidence interval
        comparison.mean_diff = comparison.mean_return_a - comparison.mean_return_b
        
        # Pooled standard error for CI
        se_diff = np.sqrt(
            (comparison.std_return_a ** 2 / len(arr_a)) +
            (comparison.std_return_b ** 2 / len(arr_b))
        )
        
        # CI using t-distribution
        alpha = 1 - confidence_level
        df = min(len(arr_a), len(arr_b)) - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        
        comparison.ci_lower = comparison.mean_diff - t_crit * se_diff
        comparison.ci_upper = comparison.mean_diff + t_crit * se_diff
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(
            ((len(arr_a) - 1) * comparison.std_return_a ** 2 +
             (len(arr_b) - 1) * comparison.std_return_b ** 2) /
            (len(arr_a) + len(arr_b) - 2)
        )
        
        if pooled_std > 0:
            comparison.cohens_d = comparison.mean_diff / pooled_std
        
        # Interpret effect size
        abs_d = abs(comparison.cohens_d)
        if abs_d < 0.2:
            comparison.effect_size_interpretation = "negligible"
        elif abs_d < 0.5:
            comparison.effect_size_interpretation = "small"
        elif abs_d < 0.8:
            comparison.effect_size_interpretation = "medium"
        else:
            comparison.effect_size_interpretation = "large"
        
        # Determine winner
        if comparison.mean_return_a > comparison.mean_return_b:
            comparison.winner = name_a
        elif comparison.mean_return_b > comparison.mean_return_a:
            comparison.winner = name_b
        else:
            comparison.winner = "tie"
        
        # Confidence in result
        min_trades = min(len(returns_a), len(returns_b))
        if comparison.is_significant and min_trades >= 30:
            comparison.confidence = "high"
        elif comparison.p_value < 0.1 and min_trades >= 20:
            comparison.confidence = "medium"
        elif min_trades < 10:
            comparison.confidence = "very_low"
        else:
            comparison.confidence = "low"
        
        return comparison


# ============================================================
# A/B Test Result
# ============================================================

@dataclass
class ABTestResult:
    """
    Results from an A/B test.
    
    Contains:
    - Per-strategy performance summaries
    - Pairwise statistical comparisons
    - Rankings
    - Trade history (optional)
    """
    
    test_id: str
    config: ABTestConfig
    start_time: datetime
    end_time: datetime
    bars_processed: int
    
    # Per-strategy results
    strategy_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Pairwise comparisons
    comparisons: List[StatisticalComparison] = field(default_factory=list)
    
    # Rankings
    ranking_by_return: List[str] = field(default_factory=list)
    ranking_by_sharpe: List[str] = field(default_factory=list)
    ranking_by_win_rate: List[str] = field(default_factory=list)
    
    # Overall winner
    overall_winner: str = ""
    winner_confidence: str = ""
    
    # Trade histories (if saved)
    trade_histories: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # Equity curves (if saved)
    equity_curves: Dict[str, List[Dict]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "config": self.config.to_dict(),
            "timing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds(),
                "bars_processed": self.bars_processed,
            },
            "strategy_results": self.strategy_results,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "rankings": {
                "by_return": self.ranking_by_return,
                "by_sharpe": self.ranking_by_sharpe,
                "by_win_rate": self.ranking_by_win_rate,
            },
            "overall": {
                "winner": self.overall_winner,
                "confidence": self.winner_confidence,
            },
        }
    
    def to_dashboard(self) -> Dict[str, Any]:
        """Generate dashboard-friendly JSON output"""
        dashboard = {
            "summary": {
                "test_id": self.test_id,
                "strategies_tested": list(self.strategy_results.keys()),
                "bars_processed": self.bars_processed,
                "overall_winner": self.overall_winner,
                "confidence": self.winner_confidence,
            },
            "performance_comparison": {},
            "head_to_head": [],
            "rankings": {
                "by_return": self.ranking_by_return,
                "by_sharpe": self.ranking_by_sharpe,
                "by_win_rate": self.ranking_by_win_rate,
            },
        }
        
        # Performance comparison table
        for name, result in self.strategy_results.items():
            dashboard["performance_comparison"][name] = {
                "total_return_pct": result.get("total_pnl_pct", 0),
                "win_rate": result.get("win_rate", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                "total_trades": result.get("total_trades", 0),
                "profit_factor": result.get("profit_factor", 0),
            }
        
        # Head-to-head comparisons
        for comp in self.comparisons:
            dashboard["head_to_head"].append({
                "matchup": f"{comp.strategy_a} vs {comp.strategy_b}",
                "winner": comp.winner,
                "p_value": round(float(comp.p_value), 4),
                "significant": bool(comp.is_significant),
                "effect_size": comp.effect_size_interpretation,
                "return_diff": round(float(comp.mean_diff), 4),
            })
        
        return dashboard
    
    def save_dashboard(self, filepath: Union[str, Path]):
        """Save dashboard JSON to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dashboard(), f, indent=2)
        
        logger.info(f"Dashboard saved to {filepath}")
    
    def save_full_results(self, filepath: Union[str, Path]):
        """Save full results including trade history"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        full_results = self.to_dict()
        full_results["trade_histories"] = self.trade_histories
        full_results["equity_curves"] = self.equity_curves
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Full results saved to {filepath}")
    
    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "=" * 70)
        print("A/B TEST RESULTS")
        print("=" * 70)
        print(f"Test ID: {self.test_id}")
        print(f"Bars processed: {self.bars_processed}")
        print(f"Strategies tested: {len(self.strategy_results)}")
        print()
        
        # Performance table
        print("PERFORMANCE COMPARISON:")
        print("-" * 70)
        print(f"{'Strategy':<20} {'Return %':>10} {'Win Rate':>10} {'Sharpe':>10} {'Trades':>8}")
        print("-" * 70)
        
        for name in self.ranking_by_return:
            r = self.strategy_results[name]
            print(
                f"{name:<20} "
                f"{r.get('total_pnl_pct', 0):>10.2f} "
                f"{r.get('win_rate', 0) * 100:>9.1f}% "
                f"{r.get('sharpe_ratio', 0):>10.2f} "
                f"{r.get('total_trades', 0):>8}"
            )
        
        print()
        
        # Statistical comparisons
        if self.comparisons:
            print("STATISTICAL COMPARISONS:")
            print("-" * 70)
            for comp in self.comparisons:
                sig = "***" if comp.is_significant else ""
                print(
                    f"{comp.strategy_a} vs {comp.strategy_b}: "
                    f"p={comp.p_value:.4f}{sig} | "
                    f"Winner: {comp.winner} ({comp.confidence} confidence)"
                )
        
        print()
        print(f"OVERALL WINNER: {self.overall_winner} ({self.winner_confidence} confidence)")
        print("=" * 70)


# ============================================================
# A/B Test Main Class
# ============================================================

class ABTest:
    """
    Main A/B testing framework.
    
    Usage:
        ab_test = ABTest(config)
        ab_test.register_strategy("strategy_a", StrategyA())
        ab_test.register_strategy("strategy_b", StrategyB())
        
        result = ab_test.run(market_data)
        result.print_summary()
        result.save_dashboard("results.json")
    """
    
    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self.test_id = str(uuid.uuid4())[:8]
        
        # Strategy registry
        self.strategies: Dict[str, Strategy] = {}
        self.runners: Dict[str, StrategyRunner] = {}
        
        # Shared data feed
        self.current_prices: Dict[str, float] = {}
        self.current_state: Optional[MarketState] = None
        
        # Progress tracking
        self.bars_processed: int = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        logger.info(f"A/B Test initialized: {self.test_id}")
    
    def register_strategy(
        self,
        name: str,
        strategy: Strategy,
        initial_weight: float = 1.0,
    ) -> bool:
        """
        Register a strategy for testing.
        
        Args:
            name: Unique name for this strategy instance
            strategy: Strategy object implementing the Strategy interface
            initial_weight: Not used in A/B testing, kept for compatibility
            
        Returns:
            True if registered successfully
        """
        if name in self.strategies:
            logger.warning(f"Strategy {name} already registered")
            return False
        
        # Create paper portfolio for this strategy
        portfolio = PaperPortfolio(
            strategy_name=name,
            initial_balance=self.config.initial_balance,
            trading_fee_pct=self.config.trading_fee_pct,
            slippage_pct=self.config.slippage_pct,
        )
        
        # Create runner
        runner = StrategyRunner(
            name=name,
            strategy=strategy,
            portfolio=portfolio,
        )
        
        self.strategies[name] = strategy
        self.runners[name] = runner
        
        logger.info(f"Registered strategy: {name}")
        return True
    
    def unregister_strategy(self, name: str) -> bool:
        """Remove a strategy from the test"""
        if name not in self.strategies:
            return False
        
        del self.strategies[name]
        del self.runners[name]
        
        logger.info(f"Unregistered strategy: {name}")
        return True
    
    def _create_market_state(
        self,
        bar: Dict[str, Any],
        indicators: Dict[str, float],
        prices_history: List[float],
    ) -> MarketState:
        """Create MarketState from bar data"""
        timestamp = bar.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return MarketState(
            symbol=bar.get("symbol", self.config.symbols[0] if self.config.symbols else "UNKNOWN"),
            timestamp=timestamp,
            open=float(bar.get("open", 0)),
            high=float(bar.get("high", 0)),
            low=float(bar.get("low", 0)),
            close=float(bar.get("close", 0)),
            volume=float(bar.get("volume", 0)),
            prices=prices_history,
            indicators=indicators,
            regime=bar.get("regime", "unknown"),
            regime_confidence=float(bar.get("regime_confidence", 0)),
            volatility=float(bar.get("volatility", indicators.get("atr", 0))),
            trend_strength=float(bar.get("trend_strength", indicators.get("adx", 0))),
        )
    
    def run(
        self,
        market_data: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ABTestResult:
        """
        Run the A/B test on market data.
        
        Args:
            market_data: List of OHLCV bars with indicators
                Each bar should have: timestamp, open, high, low, close, volume
                And indicator fields: ema_fast, ema_slow, rsi, macd, etc.
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            ABTestResult with all comparisons and statistics
        """
        if not self.strategies:
            raise ValueError("No strategies registered. Use register_strategy() first.")
        
        if not market_data:
            raise ValueError("No market data provided.")
        
        self.start_time = datetime.now()
        
        # Determine test duration
        total_bars = len(market_data)
        if self.config.test_duration_bars:
            total_bars = min(total_bars, self.config.test_duration_bars + self.config.warmup_bars)
        
        start_bar = self.config.warmup_bars
        prices_history: List[float] = []
        
        logger.info(f"Starting A/B test with {len(self.strategies)} strategies, {total_bars} bars")
        
        # Initialize strategies
        for name, strategy in self.strategies.items():
            try:
                strategy.initialize(market_data[:start_bar])
            except Exception as e:
                logger.warning(f"Strategy {name} initialization failed: {e}")
        
        # Process each bar
        for i in range(total_bars):
            bar = market_data[i]
            
            # Extract price
            close_price = float(bar.get("close", 0))
            prices_history.append(close_price)
            if len(prices_history) > 100:
                prices_history = prices_history[-100:]
            
            # Skip warmup period
            if i < start_bar:
                continue
            
            # Get timestamp
            timestamp = bar.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            
            # Build indicators dict
            indicators = {
                k: float(v) for k, v in bar.items()
                if isinstance(v, (int, float)) and k not in ["open", "high", "low", "close", "volume"]
            }
            
            # Create market state (same for all strategies - fair comparison)
            symbol = bar.get("symbol", self.config.symbols[0] if self.config.symbols else "UNKNOWN")
            state = self._create_market_state(bar, indicators, prices_history.copy())
            
            # Current prices dict
            prices = {symbol: close_price}
            
            # Run each strategy
            for name, runner in self.runners.items():
                try:
                    runner.process_bar(state, prices, timestamp)
                except Exception as e:
                    logger.error(f"Strategy {name} failed on bar {i}: {e}")
                
                # Check max drawdown stop
                if runner.portfolio.max_drawdown_pct >= self.config.max_drawdown_stop:
                    logger.warning(f"Strategy {name} hit max drawdown stop: {runner.portfolio.max_drawdown_pct:.2f}%")
            
            self.bars_processed = i - start_bar + 1
            
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i - start_bar, total_bars - start_bar)
        
        self.end_time = datetime.now()
        
        # Build results
        return self._build_result()
    
    def _build_result(self) -> ABTestResult:
        """Build ABTestResult from runners"""
        result = ABTestResult(
            test_id=self.test_id,
            config=self.config,
            start_time=self.start_time or datetime.now(),
            end_time=self.end_time or datetime.now(),
            bars_processed=self.bars_processed,
        )
        
        # Collect strategy results
        for name, runner in self.runners.items():
            result.strategy_results[name] = runner.get_summary()
            
            # Save trade history if configured
            if self.config.save_trade_history:
                result.trade_histories[name] = [
                    t.to_dict() for t in runner.portfolio.closed_trades
                ]
            
            # Save equity curve if configured
            if self.config.save_equity_curve:
                result.equity_curves[name] = runner.portfolio.equity_curve
        
        # Rankings
        result.ranking_by_return = sorted(
            result.strategy_results.keys(),
            key=lambda n: result.strategy_results[n].get("total_pnl_pct", 0),
            reverse=True
        )
        
        result.ranking_by_sharpe = sorted(
            result.strategy_results.keys(),
            key=lambda n: result.strategy_results[n].get("sharpe_ratio", 0),
            reverse=True
        )
        
        result.ranking_by_win_rate = sorted(
            result.strategy_results.keys(),
            key=lambda n: result.strategy_results[n].get("win_rate", 0),
            reverse=True
        )
        
        # Pairwise statistical comparisons
        strategy_names = list(self.runners.keys())
        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                name_a = strategy_names[i]
                name_b = strategy_names[j]
                
                runner_a = self.runners[name_a]
                runner_b = self.runners[name_b]
                
                comparison = StatisticalComparison.compare(
                    name_a=name_a,
                    returns_a=runner_a.portfolio.get_trade_returns(),
                    sharpe_a=runner_a.portfolio.calculate_sharpe_ratio(),
                    name_b=name_b,
                    returns_b=runner_b.portfolio.get_trade_returns(),
                    sharpe_b=runner_b.portfolio.calculate_sharpe_ratio(),
                    confidence_level=self.config.confidence_level,
                )
                
                result.comparisons.append(comparison)
        
        # Determine overall winner
        if result.ranking_by_sharpe:
            result.overall_winner = result.ranking_by_sharpe[0]
            
            # Confidence based on comparisons
            wins_significant = sum(
                1 for c in result.comparisons
                if c.winner == result.overall_winner and c.is_significant
            )
            total_comparisons = len(result.comparisons)
            
            if total_comparisons == 0:
                result.winner_confidence = "no_comparison"
            elif wins_significant == total_comparisons:
                result.winner_confidence = "high"
            elif wins_significant > total_comparisons / 2:
                result.winner_confidence = "medium"
            else:
                result.winner_confidence = "low"
        
        return result


# ============================================================
# Convenience Function
# ============================================================

def run_ab_test(
    strategies: Dict[str, Strategy],
    market_data: List[Dict[str, Any]],
    config: Optional[ABTestConfig] = None,
    output_path: Optional[str] = None,
) -> ABTestResult:
    """
    Convenience function to run an A/B test.
    
    Args:
        strategies: Dict of name -> Strategy
        market_data: List of OHLCV bars with indicators
        config: Optional ABTestConfig
        output_path: Optional path to save results JSON
        
    Returns:
        ABTestResult
        
    Example:
        from bot.strategy_interface import EMACrossoverStrategy, MomentumStrategy
        
        result = run_ab_test(
            strategies={
                "ema": EMACrossoverStrategy(),
                "momentum": MomentumStrategy(),
            },
            market_data=bars,
            output_path="ab_results.json"
        )
    """
    ab_test = ABTest(config)
    
    for name, strategy in strategies.items():
        ab_test.register_strategy(name, strategy)
    
    result = ab_test.run(market_data)
    
    if output_path:
        result.save_dashboard(output_path)
    
    return result


# ============================================================
# Example Strategies for Testing
# ============================================================

class SimpleMovingAverageStrategy(Strategy):
    """Simple SMA crossover for testing"""
    
    @property
    def name(self) -> str:
        return "sma_crossover"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def suitable_regimes(self) -> List[str]:
        return ["trending"]
    
    def predict(self, state: MarketState) -> StrategySignal:
        # Simple SMA logic using EMA as proxy
        ema_fast = state.indicators.get("ema_fast", state.indicators.get("ema_12", 0))
        ema_slow = state.indicators.get("ema_slow", state.indicators.get("ema_26", 0))
        
        if ema_fast > ema_slow:
            return StrategySignal(
                action=StrategyAction.BUY,
                direction="long",
                confidence=0.6,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                position_size_pct=5.0,
                reasoning="SMA bullish crossover",
            )
        elif ema_fast < ema_slow:
            return StrategySignal(
                action=StrategyAction.SELL,
                direction="short",
                confidence=0.6,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                position_size_pct=5.0,
                reasoning="SMA bearish crossover",
            )
        else:
            return StrategySignal(
                action=StrategyAction.HOLD,
                direction="flat",
                confidence=0.0,
                reasoning="No signal",
            )


class RSIStrategy(Strategy):
    """RSI-based strategy for testing"""
    
    @property
    def name(self) -> str:
        return "rsi_extremes"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def suitable_regimes(self) -> List[str]:
        return ["ranging", "sideways"]
    
    def predict(self, state: MarketState) -> StrategySignal:
        rsi = state.indicators.get("rsi", state.indicators.get("rsi_14", 50))
        
        if rsi < 30:
            return StrategySignal(
                action=StrategyAction.BUY,
                direction="long",
                confidence=min(0.9, 0.5 + (30 - rsi) / 60),
                stop_loss_pct=1.5,
                take_profit_pct=3.0,
                position_size_pct=5.0,
                reasoning=f"RSI oversold: {rsi:.1f}",
            )
        elif rsi > 70:
            return StrategySignal(
                action=StrategyAction.SELL,
                direction="short",
                confidence=min(0.9, 0.5 + (rsi - 70) / 60),
                stop_loss_pct=1.5,
                take_profit_pct=3.0,
                position_size_pct=5.0,
                reasoning=f"RSI overbought: {rsi:.1f}",
            )
        else:
            return StrategySignal(
                action=StrategyAction.HOLD,
                direction="flat",
                confidence=0.0,
                reasoning=f"RSI neutral: {rsi:.1f}",
            )


__all__ = [
    "ABTest",
    "ABTestConfig",
    "ABTestResult",
    "PaperPortfolio",
    "StrategyRunner",
    "StatisticalComparison",
    "Trade",
    "run_ab_test",
    "SimpleMovingAverageStrategy",
    "RSIStrategy",
]
