#!/usr/bin/env python3
"""
TSLA Paper Trading - V6 Improved Model

Runs paper trading specifically for TSLA using the validated v6 model
with 55% walk-forward accuracy.

Usage:
    python run_tsla_paper.py                # Start paper trading
    python run_tsla_paper.py --status       # Check status
    python run_tsla_paper.py --stop         # Stop trading
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Configuration
TSLA_CONFIG = {
    "symbol": "TSLA",
    "model_version": "v6_improved",
    "walk_forward_accuracy": 0.551,  # 55.1% from validation
    "confidence_threshold": 0.55,    # Match model training threshold
    "position_size_pct": 0.10,       # 10% of capital per trade
    "max_positions": 1,              # Only TSLA
    "stop_loss_pct": 0.015,          # 1.5% stop loss
    "take_profit_pct": 0.03,         # 3% take profit
    "trailing_stop_pct": 0.01,       # 1% trailing stop
    "loop_interval_seconds": 60,     # Check every minute
    "initial_capital": 10000,        # Paper trading capital
}


def setup_logging(log_dir: Path) -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tsla_paper_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


class TSLAPaperTrader:
    """
    TSLA-focused paper trader using the v6 improved model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.state_file = Path("data/tsla_paper/state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Trading state
        self.balance = config["initial_capital"]
        self.position = None
        self.trades = []
        self.total_pnl = 0.0
        
        # Model components
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_engineer = None
        
    def load_model(self) -> bool:
        """Load the TSLA v6 model."""
        try:
            import joblib
            from bot.ml.v6_feature_extractor import build_v6_features
            
            model_dir = Path("data/models")
            
            # Load model
            model_path = model_dir / "TSLA_xgboost_model.pkl"
            if not model_path.exists():
                self.logger.error(f"Model not found: {model_path}")
                return False
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = model_dir / "TSLA_xgboost_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load features
            features_path = model_dir / "TSLA_selected_features.json"
            if features_path.exists():
                with open(features_path) as f:
                    self.features = json.load(f)
            
            # Store the feature extractor function
            self.feature_extractor = build_v6_features
            
            self.logger.info(f"Loaded TSLA v6 model (VotingClassifier)")
            self.logger.info(f"Features: {len(self.features) if self.features else 'default'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    async def get_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetch current TSLA market data."""
        try:
            import yfinance as yf
            import pandas as pd
            
            # Fetch 1 month of hourly data for proper indicator calculation
            ticker = yf.Ticker("TSLA")
            df = ticker.history(period="1mo", interval="1h")
            
            if df.empty or len(df) < 50:
                self.logger.warning("Insufficient market data available")
                return None
            
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            return {
                "price": df["close"].iloc[-1],
                "data": df,
                "timestamp": datetime.now(timezone.utc),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            return None
    
    def generate_features(self, df) -> Optional[Dict[str, float]]:
        """Generate features for prediction."""
        try:
            import pandas as pd
            import numpy as np
            
            # Use v6 feature extractor (matches training)
            features_df = self.feature_extractor(df, pred_horizon=8, asset_class="stock")
            
            # Drop NaN rows
            features_df = features_df.dropna()
            
            if features_df.empty:
                return None
            
            # Get last row
            last_row = features_df.iloc[-1]
            
            # Select required features
            if self.features:
                available = [f for f in self.features if f in last_row.index]
                missing = [f for f in self.features if f not in last_row.index]
                
                if missing and len(missing) <= 5:
                    self.logger.debug(f"Missing features: {missing}")
                
                if len(available) < len(self.features) * 0.8:
                    self.logger.warning(f"Only {len(available)}/{len(self.features)} features available")
                    
                return {f: float(last_row[f]) for f in available if not pd.isna(last_row[f])}
            
            return last_row.to_dict()
            
        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate prediction from model."""
        try:
            import numpy as np
            import pandas as pd
            
            # Prepare feature vector
            if self.features:
                feature_names = [f for f in self.features if f in features]
                X = np.array([[features[f] for f in feature_names]])
            else:
                X = np.array([list(features.values())])
            
            # Scale if scaler available
            if self.scaler is not None:
                try:
                    X = self.scaler.transform(X)
                except Exception as e:
                    self.logger.debug(f"Scaling failed: {e}")
            
            # Predict
            pred = self.model.predict(X)[0]
            
            # Get probability if available
            confidence = 0.5
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                confidence = max(proba)
            
            return {
                "prediction": int(pred),  # 1 = up, 0 = down
                "confidence": float(confidence),
                "signal": "BUY" if pred == 1 else "SELL",
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"prediction": None, "confidence": 0.0, "signal": "HOLD"}
    
    def should_trade(self, prediction: Dict[str, Any], price: float) -> Optional[str]:
        """Determine if we should enter/exit a trade."""
        confidence = prediction["confidence"]
        signal = prediction["signal"]
        threshold = self.config["confidence_threshold"]
        
        # Check confidence threshold
        if confidence < threshold:
            return None
        
        # Current position check
        if self.position is None:
            # No position - can enter
            if signal == "BUY":
                return "ENTER_LONG"
        else:
            # Have position - check exit conditions
            entry_price = self.position["entry_price"]
            pnl_pct = (price - entry_price) / entry_price
            
            # Stop loss
            if pnl_pct < -self.config["stop_loss_pct"]:
                return "EXIT_STOP_LOSS"
            
            # Take profit
            if pnl_pct > self.config["take_profit_pct"]:
                return "EXIT_TAKE_PROFIT"
            
            # Signal reversal
            if signal == "SELL" and confidence > threshold:
                return "EXIT_SIGNAL"
        
        return None
    
    def execute_trade(self, action: str, price: float) -> None:
        """Execute a paper trade."""
        timestamp = datetime.now(timezone.utc)
        
        if action == "ENTER_LONG":
            # Calculate position size
            position_value = self.balance * self.config["position_size_pct"]
            shares = position_value / price
            
            self.position = {
                "symbol": "TSLA",
                "entry_price": price,
                "shares": shares,
                "entry_time": timestamp.isoformat(),
            }
            self.logger.info(f"📈 ENTERED LONG: {shares:.2f} shares @ ${price:.2f}")
            
        elif action.startswith("EXIT"):
            if self.position:
                entry_price = self.position["entry_price"]
                shares = self.position["shares"]
                pnl = (price - entry_price) * shares
                pnl_pct = (price - entry_price) / entry_price * 100
                
                self.balance += pnl
                self.total_pnl += pnl
                
                trade = {
                    "symbol": "TSLA",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "shares": shares,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": action,
                    "timestamp": timestamp.isoformat(),
                }
                self.trades.append(trade)
                
                emoji = "✅" if pnl > 0 else "❌"
                self.logger.info(
                    f"{emoji} {action}: ${pnl:.2f} ({pnl_pct:.2f}%) | "
                    f"Balance: ${self.balance:.2f}"
                )
                
                self.position = None
    
    def save_state(self) -> None:
        """Save current state to file."""
        state = {
            "balance": self.balance,
            "initial_capital": self.config["initial_capital"],
            "position": self.position,
            "total_pnl": self.total_pnl,
            "total_trades": len(self.trades),
            "winning_trades": sum(1 for t in self.trades if t["pnl"] > 0),
            "trades": self.trades[-10:],  # Last 10 trades
            "last_update": datetime.now(timezone.utc).isoformat(),
            "model_version": self.config["model_version"],
        }
        
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> bool:
        """Load state from file."""
        if not self.state_file.exists():
            return False
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            self.balance = state.get("balance", self.config["initial_capital"])
            self.position = state.get("position")
            self.total_pnl = state.get("total_pnl", 0.0)
            self.trades = state.get("trades", [])
            
            self.logger.info(f"Loaded state: Balance=${self.balance:.2f}, PnL=${self.total_pnl:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    async def trading_loop(self) -> None:
        """Main trading loop."""
        iteration = 0
        
        while self.running:
            iteration += 1
            
            try:
                # Get market data
                market = await self.get_market_data()
                if not market:
                    await asyncio.sleep(self.config["loop_interval_seconds"])
                    continue
                
                price = market["price"]
                
                # Generate features
                features = self.generate_features(market["data"])
                if not features:
                    await asyncio.sleep(self.config["loop_interval_seconds"])
                    continue
                
                # Get prediction
                prediction = self.predict(features)
                
                # Log status periodically
                if iteration % 10 == 0:
                    pos_str = f"LONG @ ${self.position['entry_price']:.2f}" if self.position else "FLAT"
                    self.logger.info(
                        f"[{iteration}] TSLA=${price:.2f} | {prediction['signal']} "
                        f"({prediction['confidence']:.1%}) | {pos_str} | Balance=${self.balance:.2f}"
                    )
                
                # Check for trade
                action = self.should_trade(prediction, price)
                if action:
                    self.execute_trade(action, price)
                    self.save_state()
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
            
            await asyncio.sleep(self.config["loop_interval_seconds"])
    
    async def start(self) -> None:
        """Start paper trading."""
        self.logger.info("=" * 60)
        self.logger.info("TSLA PAPER TRADING - V6 IMPROVED MODEL")
        self.logger.info("=" * 60)
        self.logger.info(f"Walk-Forward Accuracy: {self.config['walk_forward_accuracy']:.1%}")
        self.logger.info(f"Confidence Threshold: {self.config['confidence_threshold']:.1%}")
        self.logger.info(f"Initial Capital: ${self.config['initial_capital']:.2f}")
        self.logger.info(f"Position Size: {self.config['position_size_pct']:.0%}")
        self.logger.info("=" * 60)
        
        # Load model
        if not self.load_model():
            self.logger.error("Failed to load model, exiting")
            return
        
        # Load previous state
        self.load_state()
        
        self.running = True
        await self.trading_loop()
    
    async def stop(self) -> None:
        """Stop paper trading."""
        self.running = False
        self.save_state()
        self.logger.info("Paper trading stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        win_rate = 0.0
        if self.trades:
            wins = sum(1 for t in self.trades if t["pnl"] > 0)
            win_rate = wins / len(self.trades) * 100
        
        return {
            "symbol": "TSLA",
            "model_version": self.config["model_version"],
            "running": self.running,
            "balance": self.balance,
            "initial_capital": self.config["initial_capital"],
            "total_pnl": self.total_pnl,
            "total_pnl_pct": (self.total_pnl / self.config["initial_capital"]) * 100,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "position": self.position,
        }


async def run_paper_trading(args) -> None:
    """Run paper trading."""
    log_dir = Path("data/tsla_paper/logs")
    setup_logging(log_dir)
    
    trader = TSLAPaperTrader(TSLA_CONFIG)
    
    # Handle signals
    loop = asyncio.get_running_loop()
    
    def handle_stop(sig):
        logging.info(f"Received {sig}, stopping...")
        asyncio.create_task(trader.stop())
    
    for sig in [signal.SIGINT, signal.SIGTERM]:
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_stop(s))
        except NotImplementedError:
            signal.signal(sig, lambda *_: handle_stop(sig))
    
    await trader.start()
    
    # Final status
    status = trader.get_status()
    logging.info("\n" + "=" * 60)
    logging.info("FINAL STATUS")
    logging.info(f"Balance: ${status['balance']:.2f}")
    logging.info(f"P&L: ${status['total_pnl']:.2f} ({status['total_pnl_pct']:.2f}%)")
    logging.info(f"Trades: {status['total_trades']} (Win Rate: {status['win_rate']:.1f}%)")
    logging.info("=" * 60)


def show_status() -> None:
    """Show current status."""
    state_file = Path("data/tsla_paper/state.json")
    
    if not state_file.exists():
        print("No TSLA paper trading state found. Start trading first.")
        return
    
    with open(state_file) as f:
        state = json.load(f)
    
    print("\n" + "=" * 60)
    print("TSLA PAPER TRADING STATUS")
    print("=" * 60)
    print(f"Model Version: {state.get('model_version', 'v6_improved')}")
    print(f"Balance: ${state['balance']:.2f}")
    print(f"Initial Capital: ${state.get('initial_capital', 10000):.2f}")
    print(f"Total P&L: ${state['total_pnl']:.2f}")
    
    initial = state.get('initial_capital', 10000)
    pnl_pct = (state['total_pnl'] / initial) * 100
    print(f"Return: {pnl_pct:.2f}%")
    
    total_trades = state.get('total_trades', 0)
    winning = state.get('winning_trades', 0)
    print(f"Trades: {total_trades}")
    if total_trades > 0:
        print(f"Win Rate: {winning/total_trades*100:.1f}%")
    
    if state.get('position'):
        pos = state['position']
        print(f"\nOpen Position: LONG @ ${pos['entry_price']:.2f}")
    else:
        print("\nNo open position")
    
    print(f"\nLast Update: {state.get('last_update', 'unknown')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TSLA Paper Trading - V6 Model")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    else:
        TSLA_CONFIG["initial_capital"] = args.capital
        asyncio.run(run_paper_trading(args))


if __name__ == "__main__":
    main()
