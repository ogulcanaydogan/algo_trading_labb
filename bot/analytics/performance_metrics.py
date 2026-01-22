"""
Advanced Performance Metrics for Trading Analysis.

This module provides comprehensive metrics that matter for trading:
- Expectancy (expected $ per trade)
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk-adjusted returns)
- Calmar Ratio (return vs max drawdown)
- Profit Factor (gross profit / gross loss)
- Win Rate, Average Win/Loss, R-Multiple

These metrics are more meaningful than raw ML accuracy for trading systems.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import statistics


@dataclass
class TradeRecord:
    """Single trade record for analysis."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_time_minutes: float = 0.0
    exit_reason: str = ""  # 'take_profit', 'stop_loss', 'signal', 'manual'

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def r_multiple(self) -> float:
        """R-multiple: how many R (risk units) was won/lost."""
        # Assuming 1R = 2% risk per trade
        risk_pct = 0.02
        if self.pnl_pct == 0:
            return 0
        return self.pnl_pct / risk_pct


@dataclass
class PerformanceMetrics:
    """Comprehensive trading performance metrics."""

    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Win/Loss metrics
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    # Key ratios
    profit_factor: float = 0.0  # gross_profit / gross_loss
    expectancy: float = 0.0  # Expected $ per trade
    expectancy_pct: float = 0.0  # Expected % per trade

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0

    # Streaks
    max_win_streak: int = 0
    max_loss_streak: int = 0
    current_streak: int = 0
    current_streak_type: str = ""  # 'win' or 'loss'

    # Time analysis
    avg_hold_time_minutes: float = 0.0
    avg_win_hold_time: float = 0.0
    avg_loss_hold_time: float = 0.0

    # R-Multiple analysis
    avg_r_multiple: float = 0.0
    total_r: float = 0.0

    # By symbol breakdown
    by_symbol: Dict[str, Dict] = field(default_factory=dict)

    # Rolling metrics (last N trades)
    rolling_win_rate_10: float = 0.0
    rolling_win_rate_20: float = 0.0
    rolling_expectancy_10: float = 0.0
    rolling_expectancy_20: float = 0.0

    # Quality score (0-100)
    quality_score: float = 0.0
    quality_grade: str = "F"

    # Timestamp
    calculated_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate expectancy (expected value per trade).

    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

    Positive expectancy = profitable system over time.
    """
    if avg_loss == 0:
        return avg_win * win_rate if win_rate > 0 else 0

    loss_rate = 1 - win_rate
    return (win_rate * avg_win) - (loss_rate * abs(avg_loss))


def calculate_profit_factor(gross_profit: float, gross_loss: float) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    > 1.0 = profitable
    > 1.5 = good
    > 2.0 = excellent
    """
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0
    return gross_profit / abs(gross_loss)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted returns).

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

    > 1.0 = acceptable
    > 2.0 = good
    > 3.0 = excellent
    """
    if len(returns) < 2:
        return 0.0

    mean_return = statistics.mean(returns)
    std_return = statistics.stdev(returns)

    if std_return == 0:
        return 0.0

    # Annualize (assuming daily returns, ~252 trading days)
    annualized_return = mean_return * 252
    annualized_std = std_return * math.sqrt(252)

    return (annualized_return - risk_free_rate) / annualized_std


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino Ratio (downside risk-adjusted returns).

    Like Sharpe but only penalizes downside volatility.
    Better for asymmetric return distributions.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = statistics.mean(returns)

    # Only negative returns for downside deviation
    negative_returns = [r for r in returns if r < 0]

    if len(negative_returns) < 2:
        return float('inf') if mean_return > 0 else 0.0

    downside_std = statistics.stdev(negative_returns)

    if downside_std == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * 252
    annualized_downside_std = downside_std * math.sqrt(252)

    return (annualized_return - risk_free_rate) / annualized_downside_std


def calculate_calmar_ratio(total_return_pct: float, max_drawdown_pct: float) -> float:
    """
    Calculate Calmar Ratio (return / max drawdown).

    Measures return relative to worst loss experienced.

    > 1.0 = acceptable
    > 3.0 = good
    > 5.0 = excellent
    """
    if max_drawdown_pct == 0:
        return float('inf') if total_return_pct > 0 else 0.0

    return total_return_pct / abs(max_drawdown_pct)


def calculate_drawdown_series(equity_curve: List[float]) -> Tuple[List[float], float, float]:
    """
    Calculate drawdown series from equity curve.

    Returns: (drawdown_series, max_drawdown, max_drawdown_pct)
    """
    if not equity_curve:
        return [], 0.0, 0.0

    peak = equity_curve[0]
    drawdowns = []
    max_dd = 0.0
    max_dd_pct = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value

        dd = peak - value
        dd_pct = (dd / peak * 100) if peak > 0 else 0

        drawdowns.append(dd_pct)

        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    return drawdowns, max_dd, max_dd_pct


def calculate_streaks(trade_results: List[bool]) -> Tuple[int, int, int, str]:
    """
    Calculate win/loss streaks.

    Returns: (max_win_streak, max_loss_streak, current_streak, current_type)
    """
    if not trade_results:
        return 0, 0, 0, ""

    max_win = 0
    max_loss = 0
    current = 0
    current_type = ""

    prev_win = None

    for is_win in trade_results:
        if prev_win is None or is_win == prev_win:
            current += 1
        else:
            current = 1

        if is_win:
            current_type = "win"
            max_win = max(max_win, current)
        else:
            current_type = "loss"
            max_loss = max(max_loss, current)

        prev_win = is_win

    return max_win, max_loss, current, current_type


def calculate_quality_score(metrics: PerformanceMetrics) -> Tuple[float, str]:
    """
    Calculate overall quality score (0-100) and grade.

    Factors:
    - Expectancy (30%)
    - Profit Factor (20%)
    - Win Rate (15%)
    - Sharpe Ratio (15%)
    - Max Drawdown (10%)
    - Consistency (10%)
    """
    score = 0.0

    # Expectancy score (30 points max)
    # Good expectancy is 0.5-2% per trade
    if metrics.expectancy_pct > 0:
        exp_score = min(30, metrics.expectancy_pct * 15)  # 2% = 30 points
        score += exp_score

    # Profit Factor score (20 points max)
    # Good PF is 1.5-3.0
    if metrics.profit_factor > 1:
        pf_score = min(20, (metrics.profit_factor - 1) * 20)  # 2.0 = 20 points
        score += pf_score

    # Win Rate score (15 points max)
    # 50-70% is good for most strategies
    if metrics.win_rate > 0.4:
        wr_score = min(15, (metrics.win_rate - 0.4) * 50)  # 70% = 15 points
        score += wr_score

    # Sharpe Ratio score (15 points max)
    # > 1.0 is acceptable, > 2.0 is good
    if metrics.sharpe_ratio > 0:
        sharpe_score = min(15, metrics.sharpe_ratio * 7.5)  # 2.0 = 15 points
        score += sharpe_score

    # Max Drawdown score (10 points max)
    # Lower is better, < 10% is good
    if metrics.max_drawdown_pct < 20:
        dd_score = max(0, 10 - metrics.max_drawdown_pct / 2)  # 0% DD = 10 points
        score += dd_score

    # Consistency score (10 points max)
    # Based on rolling metrics stability
    if metrics.total_trades >= 20:
        roll_diff = abs(metrics.rolling_win_rate_10 - metrics.rolling_win_rate_20)
        consistency = max(0, 10 - roll_diff * 50)  # 0% diff = 10 points
        score += consistency

    # Cap at 100
    score = min(100, max(0, score))

    # Grade
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "F"

    return score, grade


def calculate_all_metrics(
    trades: List[Dict],
    equity_curve: Optional[List[Dict]] = None,
    initial_capital: float = 10000.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics from trade history.

    Args:
        trades: List of trade dicts with pnl, pnl_pct, etc.
        equity_curve: Optional equity curve for drawdown analysis
        initial_capital: Starting capital for calculations

    Returns:
        PerformanceMetrics with all calculated values
    """
    metrics = PerformanceMetrics()
    metrics.calculated_at = datetime.now().isoformat()

    if not trades:
        return metrics

    # Filter closed trades (those with pnl)
    closed_trades = [t for t in trades if 'pnl' in t and t.get('pnl') is not None]

    if not closed_trades:
        return metrics

    # Basic counts
    metrics.total_trades = len(closed_trades)

    pnls = [t.get('pnl', 0) or 0 for t in closed_trades]
    pnl_pcts = [t.get('pnl_pct', 0) or 0 for t in closed_trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    metrics.winning_trades = len(winners)
    metrics.losing_trades = len(losers)
    metrics.breakeven_trades = metrics.total_trades - metrics.winning_trades - metrics.losing_trades

    # Win/Loss rates
    metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
    metrics.loss_rate = metrics.losing_trades / metrics.total_trades if metrics.total_trades > 0 else 0

    # P&L totals
    metrics.total_pnl = sum(pnls)
    metrics.total_pnl_pct = sum(pnl_pcts)
    metrics.gross_profit = sum(winners) if winners else 0
    metrics.gross_loss = sum(losers) if losers else 0

    # Averages
    metrics.avg_win = statistics.mean(winners) if winners else 0
    metrics.avg_loss = statistics.mean(losers) if losers else 0
    metrics.avg_trade = statistics.mean(pnls) if pnls else 0

    winner_pcts = [t.get('pnl_pct', 0) or 0 for t in closed_trades if (t.get('pnl', 0) or 0) > 0]
    loser_pcts = [t.get('pnl_pct', 0) or 0 for t in closed_trades if (t.get('pnl', 0) or 0) < 0]

    metrics.avg_win_pct = statistics.mean(winner_pcts) if winner_pcts else 0
    metrics.avg_loss_pct = statistics.mean(loser_pcts) if loser_pcts else 0

    # Key ratios
    metrics.profit_factor = calculate_profit_factor(metrics.gross_profit, metrics.gross_loss)
    metrics.expectancy = calculate_expectancy(metrics.win_rate, metrics.avg_win, metrics.avg_loss)
    metrics.expectancy_pct = calculate_expectancy(metrics.win_rate, metrics.avg_win_pct, metrics.avg_loss_pct)

    # Risk-adjusted returns
    if len(pnl_pcts) >= 2:
        metrics.sharpe_ratio = calculate_sharpe_ratio(pnl_pcts)
        metrics.sortino_ratio = calculate_sortino_ratio(pnl_pcts)

    # Drawdown analysis
    if equity_curve:
        equity_values = [e.get('total_equity', e.get('balance', 0)) for e in equity_curve]
        if equity_values:
            dd_series, max_dd, max_dd_pct = calculate_drawdown_series(equity_values)
            metrics.max_drawdown = max_dd
            metrics.max_drawdown_pct = max_dd_pct
            metrics.current_drawdown_pct = dd_series[-1] if dd_series else 0
            metrics.avg_drawdown = statistics.mean(dd_series) if dd_series else 0

            # Calmar ratio
            metrics.calmar_ratio = calculate_calmar_ratio(metrics.total_pnl_pct, metrics.max_drawdown_pct)

    # Streaks
    trade_results = [p > 0 for p in pnls]
    max_win, max_loss, current, current_type = calculate_streaks(trade_results)
    metrics.max_win_streak = max_win
    metrics.max_loss_streak = max_loss
    metrics.current_streak = current
    metrics.current_streak_type = current_type

    # Hold time analysis
    hold_times = [t.get('hold_time_minutes', 0) or 0 for t in closed_trades]
    if hold_times:
        metrics.avg_hold_time_minutes = statistics.mean(hold_times)

        win_hold = [t.get('hold_time_minutes', 0) or 0 for t in closed_trades if (t.get('pnl', 0) or 0) > 0]
        loss_hold = [t.get('hold_time_minutes', 0) or 0 for t in closed_trades if (t.get('pnl', 0) or 0) < 0]

        metrics.avg_win_hold_time = statistics.mean(win_hold) if win_hold else 0
        metrics.avg_loss_hold_time = statistics.mean(loss_hold) if loss_hold else 0

    # R-Multiple analysis (assuming 2% risk per trade)
    r_multiples = [p / 2.0 for p in pnl_pcts]  # pnl_pct / risk_pct
    if r_multiples:
        metrics.avg_r_multiple = statistics.mean(r_multiples)
        metrics.total_r = sum(r_multiples)

    # By symbol breakdown
    symbols = set(t.get('symbol', 'unknown') for t in closed_trades)
    for symbol in symbols:
        sym_trades = [t for t in closed_trades if t.get('symbol') == symbol]
        sym_pnls = [t.get('pnl', 0) or 0 for t in sym_trades]
        sym_wins = len([p for p in sym_pnls if p > 0])

        metrics.by_symbol[symbol] = {
            'trades': len(sym_trades),
            'wins': sym_wins,
            'win_rate': sym_wins / len(sym_trades) if sym_trades else 0,
            'total_pnl': sum(sym_pnls),
            'avg_pnl': statistics.mean(sym_pnls) if sym_pnls else 0
        }

    # Rolling metrics
    if metrics.total_trades >= 10:
        last_10 = pnls[-10:]
        last_10_wins = len([p for p in last_10 if p > 0])
        metrics.rolling_win_rate_10 = last_10_wins / 10
        metrics.rolling_expectancy_10 = statistics.mean(last_10)

    if metrics.total_trades >= 20:
        last_20 = pnls[-20:]
        last_20_wins = len([p for p in last_20 if p > 0])
        metrics.rolling_win_rate_20 = last_20_wins / 20
        metrics.rolling_expectancy_20 = statistics.mean(last_20)

    # Quality score
    metrics.quality_score, metrics.quality_grade = calculate_quality_score(metrics)

    return metrics


def load_trades_from_file(filepath: str) -> List[Dict]:
    """Load trades from JSON file."""
    path = Path(filepath)
    if not path.exists():
        return []

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def load_equity_from_file(filepath: str) -> List[Dict]:
    """Load equity curve from JSON file."""
    path = Path(filepath)
    if not path.exists():
        return []

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


# Convenience function for quick analysis
def analyze_trading_performance(
    trades_file: str = "data/unified_trading/trades.json",
    equity_file: str = "data/unified_trading/equity.json",
    initial_capital: float = 10000.0
) -> PerformanceMetrics:
    """
    Quick function to analyze trading performance from files.
    """
    trades = load_trades_from_file(trades_file)
    equity = load_equity_from_file(equity_file)

    return calculate_all_metrics(trades, equity, initial_capital)


if __name__ == "__main__":
    # Test with sample data
    sample_trades = [
        {"symbol": "BTC/USDT", "pnl": 150, "pnl_pct": 1.5, "hold_time_minutes": 45},
        {"symbol": "BTC/USDT", "pnl": -80, "pnl_pct": -0.8, "hold_time_minutes": 30},
        {"symbol": "ETH/USDT", "pnl": 200, "pnl_pct": 2.0, "hold_time_minutes": 60},
        {"symbol": "BTC/USDT", "pnl": 120, "pnl_pct": 1.2, "hold_time_minutes": 50},
        {"symbol": "ETH/USDT", "pnl": -100, "pnl_pct": -1.0, "hold_time_minutes": 25},
        {"symbol": "BTC/USDT", "pnl": 180, "pnl_pct": 1.8, "hold_time_minutes": 55},
    ]

    metrics = calculate_all_metrics(sample_trades)

    print("=== Trading Performance Metrics ===\n")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Expectancy: ${metrics.expectancy:.2f} per trade")
    print(f"Expectancy %: {metrics.expectancy_pct:.2f}% per trade")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Quality Score: {metrics.quality_score:.1f} ({metrics.quality_grade})")
    print(f"\nBy Symbol:")
    for sym, data in metrics.by_symbol.items():
        print(f"  {sym}: {data['trades']} trades, {data['win_rate']:.1%} win rate, ${data['total_pnl']:.2f} P&L")
