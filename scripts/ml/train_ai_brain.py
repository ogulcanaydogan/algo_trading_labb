#!/usr/bin/env python3
"""
Train AI Brain from Historical Data

This script feeds historical market data to the AI Trading Brain's
Pattern Learner so it can learn what conditions lead to profits.

Usage:
    python scripts/ml/train_ai_brain.py --symbol BTC/USDT --days 365
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.ai_trading_brain import (
    get_ai_brain,
    MarketSnapshot,
    MarketCondition,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_volatility_percentile(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility percentile."""
    volatility = returns.rolling(window=window).std() * np.sqrt(24)  # Annualized for hourly
    vol_percentile = volatility.rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    return vol_percentile.fillna(50)


def determine_market_condition(
    price: float,
    sma_20: float,
    sma_50: float,
    rsi: float,
    volatility_pct: float
) -> MarketCondition:
    """Determine market condition from indicators."""
    # Price vs moving averages
    above_sma20 = price > sma_20
    above_sma50 = price > sma_50

    # Trend strength
    sma_trend = (sma_20 - sma_50) / sma_50 * 100 if sma_50 > 0 else 0

    # High volatility overrides
    if volatility_pct > 90:
        return MarketCondition.VOLATILE

    # Strong trends
    if above_sma20 and above_sma50 and sma_trend > 5:
        return MarketCondition.STRONG_BULL
    elif above_sma20 and above_sma50 and sma_trend > 2:
        return MarketCondition.BULL
    elif above_sma20 and above_sma50:
        return MarketCondition.WEAK_BULL
    elif not above_sma20 and not above_sma50 and sma_trend < -5:
        return MarketCondition.STRONG_BEAR
    elif not above_sma20 and not above_sma50 and sma_trend < -2:
        return MarketCondition.BEAR
    elif not above_sma20 and not above_sma50:
        return MarketCondition.WEAK_BEAR
    else:
        return MarketCondition.SIDEWAYS


def determine_trend(returns_4h: float) -> str:
    """Determine trend direction from recent returns."""
    if returns_4h > 1:
        return "up"
    elif returns_4h < -1:
        return "down"
    return "neutral"


async def load_historical_data(symbol: str, days: int) -> pd.DataFrame:
    """Load historical OHLCV data."""
    try:
        import ccxt

        exchange = ccxt.binance({
            'enableRateLimit': True,
        })

        # Calculate timeframe
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        logger.info(f"Fetching {days} days of data for {symbol}...")

        all_ohlcv = []
        current_since = since

        while True:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe='1h',
                since=current_since,
                limit=1000
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1

            # Check if we've reached current time
            if current_since > datetime.now().timestamp() * 1000:
                break

            logger.info(f"  Fetched {len(all_ohlcv)} candles...")

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        logger.info(f"Loaded {len(df)} hourly candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def prepare_training_data(df: pd.DataFrame, symbol: str) -> list:
    """Prepare training data with market snapshots and future returns."""
    if df.empty:
        return []

    # Calculate indicators
    df['returns'] = df['close'].pct_change() * 100
    df['rsi'] = calculate_rsi(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility_pct'] = calculate_volatility_percentile(df['returns'])

    # Calculate future returns (what we're trying to predict)
    df['return_1h'] = df['close'].shift(-1) / df['close'] * 100 - 100
    df['return_4h'] = df['close'].shift(-4) / df['close'] * 100 - 100
    df['return_24h'] = df['close'].shift(-24) / df['close'] * 100 - 100

    # Calculate recent returns for trend
    df['return_past_4h'] = df['close'] / df['close'].shift(4) * 100 - 100

    # Drop NaN rows
    df = df.dropna()

    training_data = []

    for idx, row in df.iterrows():
        # Determine market condition
        condition = determine_market_condition(
            row['close'],
            row['sma_20'],
            row['sma_50'],
            row['rsi'],
            row['volatility_pct']
        )

        # Determine trend
        trend = determine_trend(row['return_past_4h'])

        # Create snapshot
        snapshot = MarketSnapshot(
            timestamp=idx.to_pydatetime(),
            symbol=symbol,
            price=row['close'],
            trend_1h=trend,
            rsi=row['rsi'],
            volatility_percentile=row['volatility_pct'],
            condition=condition,
            volume_ratio=1.0,  # Simplified
            distance_to_support=2.0,  # Simplified
            distance_to_resistance=2.0,
        )

        training_data.append({
            'snapshot': snapshot,
            'return_1h': row['return_1h'],
            'return_4h': row['return_4h'],
            'return_24h': row['return_24h'],
        })

    return training_data


async def train_pattern_learner(symbol: str, days: int):
    """Train the AI Brain pattern learner from historical data."""
    logger.info(f"=== Training AI Brain for {symbol} ({days} days) ===")

    # Load data
    df = await load_historical_data(symbol, days)
    if df.empty:
        logger.error("No data loaded, aborting")
        return

    # Prepare training data
    training_data = prepare_training_data(df, symbol)
    logger.info(f"Prepared {len(training_data)} training samples")

    # Get AI Brain
    brain = get_ai_brain()

    # Feed to pattern learner
    learned_count = 0
    for i, sample in enumerate(training_data):
        try:
            brain.pattern_learner.learn_from_movement(
                snapshot=sample['snapshot'],
                next_1h_return=sample['return_1h'],
                next_4h_return=sample['return_4h'],
                next_24h_return=sample['return_24h']
            )
            learned_count += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{len(training_data)} samples...")

        except Exception as e:
            logger.warning(f"Error learning sample {i}: {e}")

    logger.info(f"Successfully learned from {learned_count} samples")

    # Show what was learned
    profitable = len(brain.pattern_learner.profitable_patterns)
    losing = len(brain.pattern_learner.losing_patterns)
    logger.info(f"Pattern stats: {profitable} profitable, {losing} losing patterns")

    # Show best conditions
    best_buy = brain.pattern_learner.get_best_conditions_for_action("buy", min_samples=10)
    best_sell = brain.pattern_learner.get_best_conditions_for_action("sell", min_samples=10)

    logger.info("\n=== Best BUY Conditions ===")
    for i, cond in enumerate(best_buy[:5]):
        logger.info(f"  {i+1}. {cond['condition']}: {cond['avg_return']:.2f}% avg return, {cond['win_rate']:.0%} win rate")

    logger.info("\n=== Best SELL Conditions ===")
    for i, cond in enumerate(best_sell[:5]):
        logger.info(f"  {i+1}. {cond['condition']}: {cond['avg_return']:.2f}% avg return, {cond['win_rate']:.0%} win rate")


async def main():
    parser = argparse.ArgumentParser(description='Train AI Brain from historical data')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data')
    args = parser.parse_args()

    await train_pattern_learner(args.symbol, args.days)

    # Optionally train on multiple symbols
    additional_symbols = ['ETH/USDT', 'SOL/USDT']
    for sym in additional_symbols:
        try:
            await train_pattern_learner(sym, args.days)
        except Exception as e:
            logger.error(f"Error training on {sym}: {e}")


if __name__ == '__main__':
    asyncio.run(main())
