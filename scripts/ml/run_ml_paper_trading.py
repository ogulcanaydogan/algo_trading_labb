#!/usr/bin/env python3
"""
ML Paper Trading - Use trained RandomForest/GradientBoosting models.

Runs paper trading using the models trained by run_ml_training.py.
Fetches real market data and generates ML-based trading signals.

Usage:
    python run_ml_paper_trading.py
    python run_ml_paper_trading.py --capital 50000 --interval 300
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
LOG_DIR = Path("data/ml_paper_trading/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ml_paper_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ml-paper-trading")

# Suppress noisy loggers
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

try:
    import yfinance as yf
except ImportError:
    logger.error("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

from bot.ml.predictor import MLPredictor, PredictionResult


# Symbol mapping (internal -> Yahoo Finance)
SYMBOL_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
}


class Position:
    """Track a single position."""
    def __init__(self, symbol: str, quantity: float, entry_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = datetime.now()

    @property
    def value(self) -> float:
        return self.quantity * self.entry_price

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
        }


class MLPaperTrader:
    """
    Paper trading engine using trained ML models.

    Features:
    - Loads trained RandomForest/GradientBoosting models
    - Fetches real-time prices from Yahoo Finance
    - Executes paper trades based on ML signals
    - Tracks P&L and positions
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 0.2,  # 20% of capital per trade
        confidence_threshold: float = 0.5,  # Minimum confidence to trade
        model_type: str = "random_forest",
        data_dir: str = "data/ml_paper_trading",
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size = position_size
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Positions and history
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.equity_history: List[Dict] = []

        # Load ML models
        self.models: Dict[str, MLPredictor] = {}
        self._load_models()

        # Price cache
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, datetime] = {}

    def _load_models(self):
        """Load trained models for each symbol."""
        for symbol in SYMBOL_MAP.keys():
            symbol_key = symbol.replace("/", "_")
            model_name = f"{symbol_key}_1h_{self.model_type}"
            model_dir = f"data/models/{model_name}"

            try:
                predictor = MLPredictor(
                    model_type=self.model_type,
                    model_dir=model_dir,
                )
                if predictor.load(model_name):
                    self.models[symbol] = predictor
                    logger.info(f"Loaded model for {symbol}: {model_name}")
                else:
                    logger.warning(f"Could not load model for {symbol}")
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")

        if not self.models:
            logger.error("No models loaded! Run training first:")
            logger.error("  python run_ml_training.py")
            sys.exit(1)

    def fetch_prices(self, symbol: str, period: str = "60d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch price data from Yahoo Finance."""
        yf_symbol = SYMBOL_MAP.get(symbol)
        if not yf_symbol:
            return None

        # Rate limiting - don't fetch more than once per minute
        now = datetime.now()
        if symbol in self._last_fetch:
            elapsed = (now - self._last_fetch[symbol]).total_seconds()
            if elapsed < 60 and symbol in self._price_cache:
                return self._price_cache[symbol]

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return self._price_cache.get(symbol)

            df.columns = [c.lower() for c in df.columns]
            result = df[["open", "high", "low", "close", "volume"]].copy()

            self._price_cache[symbol] = result
            self._last_fetch[symbol] = now

            return result

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return self._price_cache.get(symbol)

    def get_prediction(self, symbol: str) -> Optional[PredictionResult]:
        """Get ML prediction for a symbol."""
        if symbol not in self.models:
            return None

        # Fetch recent price data
        df = self.fetch_prices(symbol)
        if df is None or len(df) < 100:
            return None

        try:
            prediction = self.models[symbol].predict(df)
            return prediction
        except Exception as e:
            logger.warning(f"Prediction failed for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        df = self.fetch_prices(symbol)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        return 0.0

    @property
    def portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            total += pos.quantity * current_price
        return total

    def execute_trade(self, symbol: str, action: str, price: float) -> Optional[Dict]:
        """Execute a paper trade."""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": 0,
            "value": 0,
            "status": "pending",
        }

        if action == "BUY":
            # Calculate position size
            trade_value = self.cash * self.position_size
            if trade_value < 10:  # Minimum trade value
                logger.info(f"  Insufficient cash for {symbol} BUY")
                return None

            quantity = trade_value / price
            self.cash -= trade_value

            # Create or add to position
            if symbol in self.positions:
                old_pos = self.positions[symbol]
                total_qty = old_pos.quantity + quantity
                avg_price = (old_pos.quantity * old_pos.entry_price + quantity * price) / total_qty
                self.positions[symbol] = Position(symbol, total_qty, avg_price)
            else:
                self.positions[symbol] = Position(symbol, quantity, price)

            trade["quantity"] = quantity
            trade["value"] = trade_value
            trade["status"] = "executed"

        elif action == "SELL":
            if symbol not in self.positions:
                logger.info(f"  No position to sell for {symbol}")
                return None

            pos = self.positions[symbol]
            trade_value = pos.quantity * price
            self.cash += trade_value

            trade["quantity"] = pos.quantity
            trade["value"] = trade_value
            trade["pnl"] = trade_value - (pos.quantity * pos.entry_price)
            trade["status"] = "executed"

            del self.positions[symbol]

        self.trade_history.append(trade)
        return trade

    def run_iteration(self) -> Dict[str, Any]:
        """Run a single trading iteration."""
        iteration_result = {
            "timestamp": datetime.now().isoformat(),
            "signals": {},
            "trades": [],
            "portfolio_value": 0,
            "cash": self.cash,
            "positions": {},
        }

        # Get predictions for each symbol
        for symbol in SYMBOL_MAP.keys():
            prediction = self.get_prediction(symbol)
            if prediction is None:
                continue

            current_price = self.get_current_price(symbol)
            has_position = symbol in self.positions

            signal_info = {
                "action": prediction.action,
                "confidence": prediction.confidence,
                "prob_long": prediction.probability_long,
                "prob_short": prediction.probability_short,
                "prob_flat": prediction.probability_flat,
                "price": current_price,
            }
            iteration_result["signals"][symbol] = signal_info

            # Log signal
            signal_emoji = "🟢" if prediction.action == "LONG" else "🔴" if prediction.action == "SHORT" else "⚪"
            logger.info(
                f"  {symbol}: ${current_price:,.2f} | {signal_emoji} {prediction.action} "
                f"({prediction.confidence:.1%}) | Position: {'Yes' if has_position else 'No'}"
            )

            # Trading logic
            if prediction.confidence >= self.confidence_threshold:
                if prediction.action == "LONG" and not has_position:
                    # Buy signal
                    trade = self.execute_trade(symbol, "BUY", current_price)
                    if trade:
                        logger.info(f"    ➡️  BUY {trade['quantity']:.6f} @ ${current_price:,.2f}")
                        iteration_result["trades"].append(trade)

                elif prediction.action == "SHORT" and has_position:
                    # Sell signal (close long position)
                    trade = self.execute_trade(symbol, "SELL", current_price)
                    if trade:
                        pnl = trade.get("pnl", 0)
                        pnl_emoji = "📈" if pnl > 0 else "📉"
                        logger.info(f"    ➡️  SELL {trade['quantity']:.6f} @ ${current_price:,.2f} | P&L: {pnl_emoji} ${pnl:,.2f}")
                        iteration_result["trades"].append(trade)

        # Update portfolio value
        iteration_result["portfolio_value"] = self.portfolio_value
        iteration_result["cash"] = self.cash

        # Update positions info
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            iteration_result["positions"][symbol] = {
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "value": pos.quantity * current_price,
                "unrealized_pnl": unrealized_pnl,
            }

        # Record equity
        self.equity_history.append({
            "timestamp": iteration_result["timestamp"],
            "value": iteration_result["portfolio_value"],
        })

        return iteration_result

    def save_state(self):
        """Save current state to disk."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "pnl": self.portfolio_value - self.initial_capital,
            "pnl_pct": (self.portfolio_value - self.initial_capital) / self.initial_capital * 100,
            "model_type": self.model_type,
            "confidence_threshold": self.confidence_threshold,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "position_count": len(self.positions),
            "trade_count": len(self.trade_history),
        }

        # Save state
        with open(self.data_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save equity curve
        with open(self.data_dir / "equity.json", "w") as f:
            json.dump(self.equity_history[-1000:], f, indent=2)  # Keep last 1000

        # Save trade history
        with open(self.data_dir / "trades.json", "w") as f:
            json.dump(self.trade_history[-100:], f, indent=2)  # Keep last 100

    def get_summary(self) -> Dict:
        """Get trading summary."""
        total_pnl = self.portfolio_value - self.initial_capital
        winning_trades = [t for t in self.trade_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get("pnl", 0) < 0]

        return {
            "initial_capital": self.initial_capital,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.initial_capital * 100,
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / max(1, len(winning_trades) + len(losing_trades)) * 100,
            "positions": len(self.positions),
            "model_type": self.model_type,
        }


def run_paper_trading(
    initial_capital: float = 10000.0,
    interval: int = 300,  # 5 minutes
    model_type: str = "random_forest",
    confidence_threshold: float = 0.5,
    max_iterations: int = 0,  # 0 = infinite
):
    """Run the paper trading loop."""
    logger.info("=" * 60)
    logger.info("ML PAPER TRADING")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Confidence Threshold: {confidence_threshold:.0%}")
    logger.info(f"Loop Interval: {interval}s")
    logger.info("=" * 60)

    trader = MLPaperTrader(
        initial_capital=initial_capital,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
    )

    logger.info(f"Loaded {len(trader.models)} models")
    logger.info("Starting trading loop... (Ctrl+C to stop)")
    logger.info("")

    iteration = 0
    try:
        while True:
            iteration += 1
            loop_start = time.time()

            logger.info(f"--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # Run trading iteration
            result = trader.run_iteration()

            # Log portfolio status
            pnl = result["portfolio_value"] - initial_capital
            pnl_pct = pnl / initial_capital * 100
            pnl_emoji = "📈" if pnl >= 0 else "📉"

            logger.info(
                f"  Portfolio: ${result['portfolio_value']:,.2f} | "
                f"Cash: ${result['cash']:,.2f} | "
                f"P&L: {pnl_emoji} ${pnl:,.2f} ({pnl_pct:+.2f}%)"
            )

            # Save state
            trader.save_state()

            # Check max iterations
            if max_iterations > 0 and iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations})")
                break

            # Wait for next iteration
            elapsed = time.time() - loop_start
            sleep_time = max(1, interval - elapsed)
            logger.info(f"  Next iteration in {sleep_time:.0f}s...")
            logger.info("")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nReceived shutdown signal")

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PAPER TRADING SESSION ENDED")
    logger.info("=" * 60)

    summary = trader.get_summary()
    pnl_emoji = "📈" if summary["total_pnl"] >= 0 else "📉"

    logger.info(f"Final Portfolio: ${summary['portfolio_value']:,.2f}")
    logger.info(f"Total P&L: {pnl_emoji} ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.2f}%)")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']:.1f}%")
    logger.info(f"Open Positions: {summary['positions']}")

    # Save final state
    trader.save_state()

    return summary


def main():
    parser = argparse.ArgumentParser(description="ML Paper Trading")
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=300,
        help="Loop interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["random_forest", "gradient_boosting"],
        default="random_forest",
        help="Model type to use (default: random_forest)"
    )
    parser.add_argument(
        "--confidence", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Max iterations (0 = infinite)"
    )

    args = parser.parse_args()

    run_paper_trading(
        initial_capital=args.capital,
        interval=args.interval,
        model_type=args.model,
        confidence_threshold=args.confidence,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
