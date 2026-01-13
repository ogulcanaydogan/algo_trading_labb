#!/usr/bin/env python3
"""
Aggressive ML Paper Trading - Forces trades based on LONG vs SHORT probability.

Ignores FLAT predictions and always takes a position based on which direction
has higher probability. Uses Gradient Boosting model by default.

Usage:
    python run_ml_paper_trading_aggressive.py
    python run_ml_paper_trading_aggressive.py --capital 10000 --interval 60
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

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
LOG_DIR = Path("data/ml_paper_trading_aggressive/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "aggressive_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("aggressive-trading")

logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

try:
    import yfinance as yf
except ImportError:
    logger.error("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

from bot.ml.predictor import MLPredictor, PredictionResult


SYMBOL_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
}


class Position:
    def __init__(self, symbol: str, quantity: float, entry_price: float, side: str):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.side = side  # "LONG" or "SHORT"
        self.entry_time = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
        }


class AggressiveMLTrader:
    """
    Aggressive paper trader that always takes a position.

    Instead of respecting FLAT predictions, it compares LONG vs SHORT
    probabilities and goes with the higher one.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 0.3,  # 30% per position
        model_type: str = "gradient_boosting",
        data_dir: str = "data/ml_paper_trading_aggressive",
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size = position_size
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.equity_history: List[Dict] = []

        self.models: Dict[str, MLPredictor] = {}
        self._load_models()

        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, datetime] = {}

    def _load_models(self):
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
            logger.error("No models loaded! Run training first.")
            sys.exit(1)

    def fetch_prices(self, symbol: str, period: str = "60d", interval: str = "1h") -> Optional[pd.DataFrame]:
        yf_symbol = SYMBOL_MAP.get(symbol)
        if not yf_symbol:
            return None

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

    def get_aggressive_signal(self, symbol: str) -> Optional[Dict]:
        """Get aggressive signal - always LONG or SHORT, never FLAT."""
        if symbol not in self.models:
            return None

        df = self.fetch_prices(symbol)
        if df is None or len(df) < 100:
            return None

        try:
            prediction = self.models[symbol].predict(df)

            # Force LONG or SHORT based on probabilities
            prob_long = prediction.probability_long
            prob_short = prediction.probability_short

            if prob_long > prob_short:
                action = "LONG"
                confidence = prob_long / (prob_long + prob_short)  # Normalize
            else:
                action = "SHORT"
                confidence = prob_short / (prob_long + prob_short)

            return {
                "action": action,
                "confidence": confidence,
                "prob_long": prob_long,
                "prob_short": prob_short,
                "prob_flat": prediction.probability_flat,
                "original_action": prediction.action,
            }
        except Exception as e:
            logger.warning(f"Prediction failed for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        df = self.fetch_prices(symbol)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        return 0.0

    @property
    def portfolio_value(self) -> float:
        total = self.cash
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            if pos.side == "LONG":
                total += pos.quantity * current_price
            else:  # SHORT
                # For shorts: profit = entry - current
                pnl = (pos.entry_price - current_price) * pos.quantity
                total += pos.quantity * pos.entry_price + pnl
        return total

    def execute_trade(self, symbol: str, action: str, price: float) -> Optional[Dict]:
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": 0,
            "value": 0,
            "status": "pending",
        }

        if action in ["LONG", "SHORT"]:
            # Close existing position if opposite side
            if symbol in self.positions:
                old_pos = self.positions[symbol]
                if old_pos.side != action:
                    # Close opposite position
                    close_trade = self._close_position(symbol, price)
                    if close_trade:
                        self.trade_history.append(close_trade)
                        logger.info(f"    ➡️  Closed {old_pos.side} position")

            # Open new position if not already in same direction
            if symbol not in self.positions:
                trade_value = self.cash * self.position_size
                if trade_value < 10:
                    return None

                quantity = trade_value / price
                self.cash -= trade_value

                self.positions[symbol] = Position(symbol, quantity, price, action)
                trade["quantity"] = quantity
                trade["value"] = trade_value
                trade["side"] = action
                trade["status"] = "executed"

        self.trade_history.append(trade)
        return trade

    def _close_position(self, symbol: str, price: float) -> Optional[Dict]:
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        if pos.side == "LONG":
            pnl = (price - pos.entry_price) * pos.quantity
            value = pos.quantity * price
        else:  # SHORT
            pnl = (pos.entry_price - price) * pos.quantity
            value = pos.quantity * pos.entry_price + pnl

        self.cash += value

        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": f"CLOSE_{pos.side}",
            "price": price,
            "quantity": pos.quantity,
            "value": value,
            "pnl": pnl,
            "status": "executed",
        }

        del self.positions[symbol]
        return trade

    def run_iteration(self) -> Dict[str, Any]:
        iteration_result = {
            "timestamp": datetime.now().isoformat(),
            "signals": {},
            "trades": [],
            "portfolio_value": 0,
            "cash": self.cash,
            "positions": {},
        }

        for symbol in SYMBOL_MAP.keys():
            signal = self.get_aggressive_signal(symbol)
            if signal is None:
                continue

            current_price = self.get_current_price(symbol)
            current_position = self.positions.get(symbol)

            signal_info = {**signal, "price": current_price}
            iteration_result["signals"][symbol] = signal_info

            # Log signal
            signal_emoji = "🟢" if signal["action"] == "LONG" else "🔴"
            pos_info = f"{current_position.side}" if current_position else "None"

            logger.info(
                f"  {symbol}: ${current_price:,.2f} | {signal_emoji} {signal['action']} "
                f"({signal['confidence']:.1%}) | Pos: {pos_info} | "
                f"[L:{signal['prob_long']:.0%} S:{signal['prob_short']:.0%} F:{signal['prob_flat']:.0%}]"
            )

            # Trading logic - always follow the signal
            if current_position is None:
                # No position - open one
                trade = self.execute_trade(symbol, signal["action"], current_price)
                if trade and trade["status"] == "executed":
                    logger.info(f"    ➡️  OPEN {signal['action']} {trade['quantity']:.6f} @ ${current_price:,.2f}")
                    iteration_result["trades"].append(trade)
            elif current_position.side != signal["action"]:
                # Position exists but signal changed - flip
                trade = self.execute_trade(symbol, signal["action"], current_price)
                if trade and trade["status"] == "executed":
                    logger.info(f"    ➡️  FLIP to {signal['action']} {trade['quantity']:.6f} @ ${current_price:,.2f}")
                    iteration_result["trades"].append(trade)

        # Update portfolio value
        iteration_result["portfolio_value"] = self.portfolio_value
        iteration_result["cash"] = self.cash

        # Update positions info
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            if pos.side == "LONG":
                unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.quantity

            iteration_result["positions"][symbol] = {
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
            }

        self.equity_history.append({
            "timestamp": iteration_result["timestamp"],
            "value": iteration_result["portfolio_value"],
        })

        # Store latest signals for dashboard display
        self.latest_signals = iteration_result["signals"]

        return iteration_result

    def save_state(self):
        state = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "pnl": self.portfolio_value - self.initial_capital,
            "pnl_pct": (self.portfolio_value - self.initial_capital) / self.initial_capital * 100,
            "model_type": self.model_type,
            "mode": "aggressive",
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "position_count": len(self.positions),
            "trade_count": len(self.trade_history),
            # Include signals for dashboard display
            "signals": {
                sym: {
                    "signal": sig.get("action"),
                    "regime": sig.get("regime", "unknown"),
                    "confidence": sig.get("confidence", 0),
                    "price": sig.get("price", 0),
                }
                for sym, sig in getattr(self, "latest_signals", {}).items()
            },
        }

        with open(self.data_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        with open(self.data_dir / "equity.json", "w") as f:
            json.dump(self.equity_history[-1000:], f, indent=2)

        with open(self.data_dir / "trades.json", "w") as f:
            json.dump(self.trade_history[-100:], f, indent=2)

    def get_summary(self) -> Dict:
        total_pnl = self.portfolio_value - self.initial_capital
        trades_with_pnl = [t for t in self.trade_history if "pnl" in t]
        winning_trades = [t for t in trades_with_pnl if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades_with_pnl if t.get("pnl", 0) < 0]

        return {
            "initial_capital": self.initial_capital,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.initial_capital * 100,
            "total_trades": len(self.trade_history),
            "closed_trades": len(trades_with_pnl),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / max(1, len(trades_with_pnl)) * 100,
            "positions": len(self.positions),
            "model_type": self.model_type,
        }


def run_aggressive_trading(
    initial_capital: float = 10000.0,
    interval: int = 60,
    model_type: str = "gradient_boosting",
    max_iterations: int = 0,
):
    logger.info("=" * 60)
    logger.info("AGGRESSIVE ML PAPER TRADING")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Mode: AGGRESSIVE (always trades)")
    logger.info(f"Loop Interval: {interval}s")
    logger.info("=" * 60)

    trader = AggressiveMLTrader(
        initial_capital=initial_capital,
        model_type=model_type,
    )

    logger.info(f"Loaded {len(trader.models)} models")
    logger.info("Starting aggressive trading... (Ctrl+C to stop)")
    logger.info("")

    iteration = 0
    try:
        while True:
            iteration += 1
            loop_start = time.time()

            logger.info(f"--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            result = trader.run_iteration()

            pnl = result["portfolio_value"] - initial_capital
            pnl_pct = pnl / initial_capital * 100
            pnl_emoji = "📈" if pnl >= 0 else "📉"

            logger.info(
                f"  Portfolio: ${result['portfolio_value']:,.2f} | "
                f"Cash: ${result['cash']:,.2f} | "
                f"P&L: {pnl_emoji} ${pnl:,.2f} ({pnl_pct:+.2f}%)"
            )

            # Show positions
            if result["positions"]:
                for sym, pos in result["positions"].items():
                    pos_emoji = "🟢" if pos["side"] == "LONG" else "🔴"
                    pnl_emoji2 = "+" if pos["unrealized_pnl"] >= 0 else ""
                    logger.info(
                        f"    {pos_emoji} {sym}: {pos['side']} | "
                        f"Entry: ${pos['entry_price']:,.2f} | "
                        f"Now: ${pos['current_price']:,.2f} | "
                        f"P&L: {pnl_emoji2}${pos['unrealized_pnl']:,.2f}"
                    )

            trader.save_state()

            if max_iterations > 0 and iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations})")
                break

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
    logger.info("AGGRESSIVE TRADING SESSION ENDED")
    logger.info("=" * 60)

    summary = trader.get_summary()
    pnl_emoji = "📈" if summary["total_pnl"] >= 0 else "📉"

    logger.info(f"Final Portfolio: ${summary['portfolio_value']:,.2f}")
    logger.info(f"Total P&L: {pnl_emoji} ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.2f}%)")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Closed Trades: {summary['closed_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']:.1f}%")
    logger.info(f"Open Positions: {summary['positions']}")

    trader.save_state()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggressive ML Paper Trading")
    parser.add_argument("--capital", "-c", type=float, default=10000.0)
    parser.add_argument("--interval", "-i", type=int, default=60)
    parser.add_argument("--model", "-m", choices=["random_forest", "gradient_boosting"], default="gradient_boosting")
    parser.add_argument("--max-iterations", type=int, default=0)

    args = parser.parse_args()

    run_aggressive_trading(
        initial_capital=args.capital,
        interval=args.interval,
        model_type=args.model,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
