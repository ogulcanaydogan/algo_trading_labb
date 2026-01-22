#!/usr/bin/env python3
"""
Walk-Forward Backtesting.

More realistic backtesting with rolling retraining:
1. Train on window [0, train_end]
2. Test on window [train_end, train_end + test_period]
3. Roll forward and repeat

This prevents look-ahead bias and simulates real-world model updates.

Usage:
    python scripts/backtest/walk_forward_backtest.py --symbol BTC/USDT
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    train_samples: int = 0
    test_samples: int = 0
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'window_id': self.window_id,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'train_accuracy': round(self.train_accuracy, 4),
            'test_accuracy': round(self.test_accuracy, 4),
            'total_trades': self.total_trades,
            'total_return': round(self.total_return, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 4),
        }


@dataclass
class WalkForwardSummary:
    """Summary of walk-forward backtest."""
    symbol: str
    model_type: str
    total_windows: int
    train_period_days: int
    test_period_days: int

    # Aggregated metrics
    avg_train_accuracy: float = 0.0
    avg_test_accuracy: float = 0.0
    total_return: float = 0.0
    avg_sharpe: float = 0.0
    max_drawdown: float = 0.0

    # Consistency
    accuracy_std: float = 0.0
    return_std: float = 0.0
    profitable_windows: int = 0

    windows: List[WalkForwardResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'total_windows': self.total_windows,
            'train_period_days': self.train_period_days,
            'test_period_days': self.test_period_days,
            'avg_train_accuracy': round(self.avg_train_accuracy, 4),
            'avg_test_accuracy': round(self.avg_test_accuracy, 4),
            'total_return': round(self.total_return, 4),
            'avg_sharpe': round(self.avg_sharpe, 2),
            'max_drawdown': round(self.max_drawdown, 4),
            'accuracy_std': round(self.accuracy_std, 4),
            'profitable_windows': self.profitable_windows,
            'windows': [w.to_dict() for w in self.windows],
        }


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data."""
    data_file = Path(f"data/training/{symbol.replace('/', '_')}_extended.parquet")

    if data_file.exists():
        df = pd.read_parquet(data_file)
        # Filter to requested days
        cutoff = datetime.now() - timedelta(days=days)
        if hasattr(df.index, 'tz_localize'):
            df = df[df.index >= cutoff]
        return df

    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})

        logger.info(f"Fetching {days} days of {symbol} data...")

        all_ohlcv = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol.replace("/", ""), '1h', since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create features for ML."""
    df = df.copy()

    close = df['close']
    returns = close.pct_change()

    # EMAs
    for p in [8, 21, 55]:
        ema = close.ewm(span=p).mean()
        df[f'ema_{p}_dist'] = (close - ema) / ema

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / close

    # Volatility
    df['volatility'] = returns.rolling(20).std()

    # Returns
    for p in [1, 3, 5, 10]:
        df[f'return_{p}'] = returns.rolling(p).sum()

    # Bollinger
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_sma) / (2 * bb_std + 1e-10)

    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

    return df, feature_cols


def create_labels(df: pd.DataFrame, horizon: int = 12, threshold: float = 0.01) -> pd.Series:
    """Create labels for classification."""
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    labels = pd.Series(1, index=df.index)  # FLAT
    labels[future_return > threshold] = 2  # UP
    labels[future_return < -threshold] = 0  # DOWN
    return labels


def simulate_trading(
    predictions: np.ndarray,
    actuals: np.ndarray,
    prices: np.ndarray,
    confidence_threshold: float = 0.6
) -> Dict:
    """Simulate trading based on predictions."""
    balance = 10000.0
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0

    equity_curve = [balance]
    trades = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i]

        # Exit logic
        if position != 0:
            # Simple exit after 12 bars
            if len(trades) > 0 and i - trades[-1].get('entry_idx', 0) >= 12:
                pnl = (price - entry_price) * position / entry_price
                balance *= (1 + pnl)
                trades[-1]['exit_price'] = price
                trades[-1]['pnl'] = pnl
                position = 0

        # Entry logic
        if position == 0:
            if pred == 2:  # UP prediction
                position = 1
                entry_price = price
                trades.append({'entry_idx': i, 'entry_price': price, 'side': 'long'})
            elif pred == 0:  # DOWN prediction
                position = -1
                entry_price = price
                trades.append({'entry_idx': i, 'entry_price': price, 'side': 'short'})

        equity_curve.append(balance)

    # Calculate metrics
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_return = (balance / 10000) - 1

    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
    else:
        sharpe = 0.0

    # Drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    winning = sum(1 for t in trades if t.get('pnl', 0) > 0)

    return {
        'total_trades': len(trades),
        'winning_trades': winning,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
    }


def run_walk_forward(
    symbol: str,
    train_period_days: int = 60,
    test_period_days: int = 14,
    step_days: int = 14,
    model_type: str = 'random_forest',
    total_days: int = 365
) -> WalkForwardSummary:
    """Run walk-forward backtest."""
    logger.info(f"Walk-forward backtest: {symbol}")
    logger.info(f"Train: {train_period_days}d, Test: {test_period_days}d, Step: {step_days}d")

    # Fetch data
    df = fetch_data(symbol, total_days + train_period_days)
    if df.empty or len(df) < 1000:
        raise ValueError(f"Insufficient data for {symbol}")

    # Create features and labels
    df, feature_cols = create_features(df)
    df['target'] = create_labels(df)
    df = df.dropna()

    logger.info(f"Data: {len(df)} samples, {len(feature_cols)} features")

    # Convert to hours
    train_hours = train_period_days * 24
    test_hours = test_period_days * 24
    step_hours = step_days * 24

    results = []
    window_id = 0

    # Walk through data
    start_idx = 0
    while start_idx + train_hours + test_hours <= len(df):
        train_end_idx = start_idx + train_hours
        test_end_idx = train_end_idx + test_hours

        # Split data
        train_df = df.iloc[start_idx:train_end_idx]
        test_df = df.iloc[train_end_idx:test_end_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values.astype(int)
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values.astype(int)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        # Simulate trading
        predictions = model.predict(X_test_scaled)
        prices = test_df['close'].values
        trading_results = simulate_trading(predictions, y_test, prices)

        # Record result
        result = WalkForwardResult(
            window_id=window_id,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            train_samples=len(train_df),
            test_samples=len(test_df),
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            total_trades=trading_results['total_trades'],
            winning_trades=trading_results['winning_trades'],
            total_return=trading_results['total_return'],
            sharpe_ratio=trading_results['sharpe_ratio'],
            max_drawdown=trading_results['max_drawdown'],
        )

        results.append(result)
        logger.info(f"Window {window_id}: Train={train_acc:.2%}, Test={test_acc:.2%}, Return={trading_results['total_return']:.2%}")

        window_id += 1
        start_idx += step_hours

    # Create summary
    summary = WalkForwardSummary(
        symbol=symbol,
        model_type=model_type,
        total_windows=len(results),
        train_period_days=train_period_days,
        test_period_days=test_period_days,
        windows=results,
    )

    if results:
        test_accs = [r.test_accuracy for r in results]
        returns = [r.total_return for r in results]

        summary.avg_train_accuracy = np.mean([r.train_accuracy for r in results])
        summary.avg_test_accuracy = np.mean(test_accs)
        summary.total_return = np.prod([1 + r for r in returns]) - 1
        summary.avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        summary.max_drawdown = max(r.max_drawdown for r in results)
        summary.accuracy_std = np.std(test_accs)
        summary.return_std = np.std(returns)
        summary.profitable_windows = sum(1 for r in results if r.total_return > 0)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Walk-forward backtesting')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--train-days', type=int, default=60, help='Training period days')
    parser.add_argument('--test-days', type=int, default=14, help='Test period days')
    parser.add_argument('--step-days', type=int, default=14, help='Step size days')
    parser.add_argument('--model', choices=['random_forest', 'gradient_boosting'], default='random_forest')
    parser.add_argument('--all-symbols', action='store_true', help='Run for all symbols')
    args = parser.parse_args()

    if args.all_symbols:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    else:
        symbols = [args.symbol]

    all_results = {}
    for symbol in symbols:
        try:
            summary = run_walk_forward(
                symbol=symbol,
                train_period_days=args.train_days,
                test_period_days=args.test_days,
                step_days=args.step_days,
                model_type=args.model,
            )
            all_results[symbol] = summary

            # Save report
            symbol_clean = symbol.replace('/', '_')
            report_path = REPORT_DIR / f"walk_forward_{symbol_clean}_{args.model}.json"
            with open(report_path, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)

            logger.info(f"\nSaved report to {report_path}")

        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD BACKTEST SUMMARY")
    logger.info("=" * 60)

    for symbol, summary in all_results.items():
        logger.info(f"\n{symbol} ({summary.model_type}):")
        logger.info(f"  Windows: {summary.total_windows}")
        logger.info(f"  Avg Test Accuracy: {summary.avg_test_accuracy:.2%} (+/- {summary.accuracy_std:.2%})")
        logger.info(f"  Total Return: {summary.total_return:.2%}")
        logger.info(f"  Avg Sharpe: {summary.avg_sharpe:.2f}")
        logger.info(f"  Max Drawdown: {summary.max_drawdown:.2%}")
        logger.info(f"  Profitable Windows: {summary.profitable_windows}/{summary.total_windows}")


if __name__ == "__main__":
    main()
