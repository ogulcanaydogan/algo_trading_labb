#!/usr/bin/env python3
"""
Advanced Backtesting Runner Script.

Run backtests on historical data for different markets and strategies.

Usage:
    python run_advanced_backtest.py --symbol AAPL --strategy momentum --days 365
    python run_advanced_backtest.py --symbol BTC-USD --strategy ml --days 180
    python run_advanced_backtest.py --market commodity --days 365
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bot.advanced_backtesting import (
    AdvancedBacktester,
    BacktestConfig,
    BacktestReport,
    SignalGenerator,
    WalkForwardAnalyzer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Symbol mappings by market
MARKET_SYMBOLS = {
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "commodity": ["GC=F", "SI=F", "CL=F"],
    "stock": ["AAPL", "MSFT", "GOOGL", "AMZN"],
}


class MomentumStrategy(SignalGenerator):
    """Momentum-based trading strategy."""

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        ema_fast: int = 12,
        ema_slow: int = 26,
    ):
        super().__init__(name="momentum")
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate momentum signal based on RSI and EMA crossover."""
        if len(data) < self.ema_slow + 5:
            return {"decision": "FLAT", "confidence": 0}

        # Calculate indicators
        close = data["close"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # EMAs
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

        # Get current values
        current_rsi = rsi.iloc[-1]
        current_ema_fast = ema_fast.iloc[-1]
        current_ema_slow = ema_slow.iloc[-1]

        # Generate signal
        if current_rsi < self.rsi_oversold and current_ema_fast > current_ema_slow:
            confidence = (self.rsi_oversold - current_rsi) / self.rsi_oversold
            return {"decision": "LONG", "confidence": min(confidence, 1.0)}

        elif current_rsi > self.rsi_overbought and current_ema_fast < current_ema_slow:
            confidence = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return {"decision": "SHORT", "confidence": min(confidence, 1.0)}

        return {"decision": "FLAT", "confidence": 0}


class MeanReversionStrategy(SignalGenerator):
    """Mean reversion strategy using Bollinger Bands."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        super().__init__(name="mean_reversion")
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate signal based on Bollinger Bands."""
        if len(data) < self.bb_period + 5:
            return {"decision": "FLAT", "confidence": 0}

        close = data["close"]

        # Bollinger Bands
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)

        # Get current values
        current_close = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        band_width = current_upper - current_lower

        # Generate signal
        if current_close < current_lower:
            # Price below lower band - oversold
            distance = (current_lower - current_close) / band_width if band_width > 0 else 0
            return {"decision": "LONG", "confidence": min(distance, 1.0)}

        elif current_close > current_upper:
            # Price above upper band - overbought
            distance = (current_close - current_upper) / band_width if band_width > 0 else 0
            return {"decision": "SHORT", "confidence": min(distance, 1.0)}

        return {"decision": "FLAT", "confidence": 0}


class TrendFollowingStrategy(SignalGenerator):
    """Trend following strategy using multiple EMAs."""

    def __init__(self):
        super().__init__(name="trend_following")
        self.ema_periods = [10, 20, 50]

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate signal based on EMA alignment."""
        if len(data) < max(self.ema_periods) + 5:
            return {"decision": "FLAT", "confidence": 0}

        close = data["close"]

        # Calculate EMAs
        emas = [close.ewm(span=p, adjust=False).mean().iloc[-1] for p in self.ema_periods]

        # Check alignment
        if emas[0] > emas[1] > emas[2]:
            # Bullish alignment
            spread = (emas[0] - emas[2]) / emas[2]
            return {"decision": "LONG", "confidence": min(spread * 10, 1.0)}

        elif emas[0] < emas[1] < emas[2]:
            # Bearish alignment
            spread = (emas[2] - emas[0]) / emas[0]
            return {"decision": "SHORT", "confidence": min(spread * 10, 1.0)}

        return {"decision": "FLAT", "confidence": 0}


class MLStrategy(SignalGenerator):
    """ML-based trading strategy using trained models."""

    def __init__(self, model_dir: str = "data/models", model_type: str = "lstm"):
        super().__init__(name=f"ml_{model_type}")
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load ML model."""
        try:
            if self.model_type == "lstm":
                from bot.ml.models.deep_learning.lstm import LSTMModel
                self.model = LSTMModel(model_dir=str(self.model_dir))
            elif self.model_type == "transformer":
                from bot.ml.models.deep_learning.transformer import TransformerModel
                self.model = TransformerModel(model_dir=str(self.model_dir))
            else:
                logger.warning(f"Unknown model type: {self.model_type}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")

    def generate_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate signal using ML model."""
        if self.model is None or not self.model.is_trained:
            return {"decision": "FLAT", "confidence": 0}

        try:
            from bot.ml.feature_engineer import FeatureEngineer

            # Extract features
            fe = FeatureEngineer()
            features_df = fe.extract_features(data)

            if len(features_df) < 60:
                return {"decision": "FLAT", "confidence": 0}

            # Get feature columns
            exclude_cols = [
                "open", "high", "low", "close", "volume",
                "target_return", "target_direction", "target_class",
            ]
            feature_cols = [c for c in features_df.columns if c not in exclude_cols]

            X = features_df[feature_cols].values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

            # Get prediction
            prediction = self.model.predict(X)

            return {
                "decision": prediction.action,
                "confidence": prediction.confidence,
            }
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
            return {"decision": "FLAT", "confidence": 0}


def fetch_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch historical data from Yahoo Finance."""
    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching {symbol} data ({days} days)...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            logger.error(f"No data returned for {symbol}")
            return None

        # Normalize columns
        df.columns = df.columns.str.lower()
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def get_strategy(name: str) -> SignalGenerator:
    """Get strategy by name."""
    strategies = {
        "momentum": MomentumStrategy(),
        "mean_reversion": MeanReversionStrategy(),
        "trend_following": TrendFollowingStrategy(),
        "ml_lstm": MLStrategy(model_type="lstm"),
        "ml_transformer": MLStrategy(model_type="transformer"),
    }
    return strategies.get(name, MomentumStrategy())


def run_single_backtest(
    symbol: str,
    strategy_name: str,
    days: int,
    config: BacktestConfig,
    output_dir: Path,
) -> Optional[BacktestReport]:
    """Run backtest for a single symbol."""
    # Fetch data
    data = fetch_data(symbol, days)
    if data is None:
        return None

    # Get strategy
    strategy = get_strategy(strategy_name)

    # Run backtest
    logger.info(f"Running backtest for {symbol} with {strategy_name} strategy...")
    backtester = AdvancedBacktester(config, strategy)
    report = backtester.run(data, symbol)

    # Save report
    report_file = output_dir / f"backtest_{symbol.replace('=', '_').replace('-', '_')}_{strategy_name}.json"
    report.save(str(report_file))

    # Print summary
    report.print_summary()

    return report


def run_market_backtest(
    market: str,
    strategy_name: str,
    days: int,
    config: BacktestConfig,
    output_dir: Path,
) -> List[BacktestReport]:
    """Run backtest for all symbols in a market."""
    symbols = MARKET_SYMBOLS.get(market, [])
    reports = []

    for symbol in symbols:
        report = run_single_backtest(symbol, strategy_name, days, config, output_dir)
        if report:
            reports.append(report)

    # Print market summary
    if reports:
        print("\n" + "=" * 70)
        print(f"MARKET SUMMARY: {market.upper()}")
        print("=" * 70)
        print(f"{'Symbol':<12} {'Return %':>10} {'Win Rate':>10} {'Sharpe':>10} {'Max DD %':>10}")
        print("-" * 70)

        for r in reports:
            print(
                f"{r.symbol:<12} "
                f"{r.metrics.total_return_pct:>10.2f} "
                f"{r.metrics.win_rate:>10.2%} "
                f"{r.metrics.sharpe_ratio:>10.2f} "
                f"{r.metrics.max_drawdown_pct:>10.2f}"
            )

        # Aggregate metrics
        avg_return = np.mean([r.metrics.total_return_pct for r in reports])
        avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in reports])
        print("-" * 70)
        print(f"{'AVERAGE':<12} {avg_return:>10.2f} {'':<10} {avg_sharpe:>10.2f}")
        print("=" * 70)

    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Run advanced backtests on historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        help="Single symbol to backtest (e.g., AAPL, BTC-USD, GC=F)",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["crypto", "commodity", "stock", "all"],
        help="Market to backtest (runs all symbols in market)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        choices=["momentum", "mean_reversion", "trend_following", "ml_lstm", "ml_transformer"],
        help="Trading strategy to use",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000,
        help="Initial account balance",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.1,
        help="Commission percentage per trade",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=2.0,
        help="Stop loss percentage",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=4.0,
        help="Take profit percentage",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtest_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward analysis",
    )

    args = parser.parse_args()

    if not args.symbol and not args.market:
        parser.error("Must specify either --symbol or --market")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = BacktestConfig(
        initial_balance=args.initial_balance,
        commission_pct=args.commission,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )

    logger.info(f"Backtest Configuration:")
    logger.info(f"  Initial Balance: ${config.initial_balance:,.2f}")
    logger.info(f"  Commission: {config.commission_pct}%")
    logger.info(f"  Stop Loss: {config.stop_loss_pct}%")
    logger.info(f"  Take Profit: {config.take_profit_pct}%")
    logger.info(f"  Strategy: {args.strategy}")

    # Run backtests
    if args.symbol:
        run_single_backtest(
            args.symbol,
            args.strategy,
            args.days,
            config,
            output_dir,
        )
    elif args.market == "all":
        for market in ["crypto", "commodity", "stock"]:
            run_market_backtest(market, args.strategy, args.days, config, output_dir)
    else:
        run_market_backtest(args.market, args.strategy, args.days, config, output_dir)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
