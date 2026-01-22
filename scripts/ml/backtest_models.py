#!/usr/bin/env python3
"""
Backtest ML Models on Historical Data.

Tests model predictions against actual price movements.
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Same 22 features
OPTIMAL_FEATURES = [
    "ema_8_dist", "ema_21_dist", "ema_55_dist", "ema_100_dist",
    "rsi", "rsi_norm", "macd", "macd_signal", "macd_hist",
    "volatility", "volatility_ratio", "volume_ratio",
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "bb_position", "bb_width", "atr", "momentum", "momentum_acc",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 22 optimal features."""
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    for period in [8, 21, 55, 100]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"ema_{period}_dist"] = (close - ema) / ema

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = 100 - (100 / (1 + rs))
    features["rsi_norm"] = (features["rsi"] - 50) / 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    features["volatility"] = close.pct_change().rolling(20).std()
    features["volatility_ratio"] = features["volatility"] / features["volatility"].rolling(50).mean()
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    for period in [1, 3, 5, 10, 20]:
        features[f"return_{period}"] = close.pct_change(period)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features["bb_position"] = (close - lower) / (upper - lower + 1e-10)
    features["bb_width"] = (upper - lower) / sma20

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / close

    features["momentum"] = close.pct_change(10)
    features["momentum_acc"] = features["momentum"] - features["momentum"].shift(5)

    return features[OPTIMAL_FEATURES]


def fetch_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch historical data."""
    import yfinance as yf

    yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
    logger.info(f"Fetching {days} days for {symbol}...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"  Got {len(df)} candles")
    return df


def load_models(symbol: str, model_dir: Path) -> Dict:
    """Load all models for a symbol."""
    symbol_safe = symbol.replace("/", "_")
    models = {}

    for model_type in ["random_forest", "gradient_boosting", "xgboost"]:
        model_path = model_dir / f"{symbol_safe}_{model_type}_model.pkl"
        scaler_path = model_dir / f"{symbol_safe}_{model_type}_scaler.pkl"
        meta_path = model_dir / f"{symbol_safe}_{model_type}_meta.json"

        if model_path.exists() and scaler_path.exists():
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                models[model_type] = {"model": model, "scaler": scaler, "accuracy": meta.get("accuracy", 0)}
                logger.info(f"  Loaded {model_type} (accuracy: {meta.get('accuracy', 0):.2%})")
            except Exception as e:
                logger.warning(f"  Failed to load {model_type}: {e}")

    return models


def backtest_strategy(
    df: pd.DataFrame,
    models: Dict,
    initial_capital: float = 10000.0,
    position_size: float = 0.2,  # 20% of capital per trade
    stop_loss: float = 0.02,  # 2%
    take_profit: float = 0.04,  # 4%
) -> Dict:
    """Run backtest with ensemble predictions."""

    X = compute_features(df)
    valid = X.notna().all(axis=1)
    X = X[valid]
    df = df[valid]

    # Get ensemble predictions
    predictions = []
    for i in range(len(X)):
        votes = []
        weights = []
        for model_type, model_data in models.items():
            try:
                X_scaled = model_data["scaler"].transform(X.iloc[i:i+1].values)
                pred = model_data["model"].predict(X_scaled)[0]
                votes.append(pred)
                weights.append(model_data["accuracy"])
            except:
                continue

        if votes:
            # Weighted voting
            weighted_votes = np.zeros(3)
            for v, w in zip(votes, weights):
                weighted_votes[int(v)] += w
            predictions.append(np.argmax(weighted_votes))
        else:
            predictions.append(1)  # HOLD

    df = df.copy()
    df["prediction"] = predictions
    df["actual_return"] = df["close"].pct_change().shift(-1)

    # Simulate trading
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]

    for i in range(len(df) - 1):
        current_price = df["close"].iloc[i]
        pred = df["prediction"].iloc[i]

        if position is None:
            # Open position
            if pred == 2:  # LONG
                position = {
                    "side": "LONG",
                    "entry_price": current_price,
                    "size": capital * position_size / current_price,
                    "entry_idx": i
                }
            elif pred == 0:  # SHORT
                position = {
                    "side": "SHORT",
                    "entry_price": current_price,
                    "size": capital * position_size / current_price,
                    "entry_idx": i
                }
        else:
            # Check exit conditions
            if position["side"] == "LONG":
                pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
            else:  # SHORT
                pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]

            should_exit = (
                pnl_pct <= -stop_loss or  # Stop loss
                pnl_pct >= take_profit or  # Take profit
                pred == 1 or  # HOLD signal
                (position["side"] == "LONG" and pred == 0) or  # Reversal
                (position["side"] == "SHORT" and pred == 2)  # Reversal
            )

            if should_exit:
                pnl = pnl_pct * position["size"] * position["entry_price"]
                capital += pnl
                trades.append({
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "duration": i - position["entry_idx"]
                })
                position = None

        equity_curve.append(capital)

    # Close any remaining position
    if position:
        current_price = df["close"].iloc[-1]
        if position["side"] == "LONG":
            pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]
        pnl = pnl_pct * position["size"] * position["entry_price"]
        capital += pnl
        trades.append({
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration": len(df) - position["entry_idx"]
        })

    # Calculate metrics
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.max(equity_curve)

    return {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return": round((capital / initial_capital - 1) * 100, 2),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]) * 100, 2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]) * 100, 2) if losses else 0,
        "max_drawdown": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24), 2),
        "profit_factor": round(sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)), 2) if losses and sum(t["pnl"] for t in losses) != 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--capital", type=float, default=10000)
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    model_dir = Path(args.model_dir)

    logger.info("=" * 60)
    logger.info("ML MODEL BACKTEST")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {args.days} days")
    logger.info(f"Initial Capital: ${args.capital}")
    logger.info("=" * 60)

    results = {}
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting {symbol}")
        logger.info("=" * 60)

        try:
            df = fetch_data(symbol, days=args.days)
            if df is None or len(df) < 500:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            models = load_models(symbol, model_dir)
            if not models:
                logger.warning(f"No models found for {symbol}")
                continue

            result = backtest_strategy(df, models, initial_capital=args.capital)
            results[symbol] = result

            logger.info(f"\nResults for {symbol}:")
            logger.info(f"  Total Return: {result['total_return']}%")
            logger.info(f"  Win Rate: {result['win_rate']}%")
            logger.info(f"  Total Trades: {result['total_trades']}")
            logger.info(f"  Max Drawdown: {result['max_drawdown']}%")
            logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']}")
            logger.info(f"  Profit Factor: {result['profit_factor']}")

        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    for symbol, result in results.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Return: {result['total_return']}% | Win Rate: {result['win_rate']}% | Sharpe: {result['sharpe_ratio']}")


if __name__ == "__main__":
    main()
