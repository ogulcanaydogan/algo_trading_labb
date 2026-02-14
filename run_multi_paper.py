#!/usr/bin/env python3
"""
Multi-Asset Paper Trading - V6 Improved Models

Trades multiple assets simultaneously:
- TSLA (stocks via Alpaca paper)
- XRP_USDT (crypto via CCXT/simulation)
- BTC_USDT (crypto via CCXT/simulation)

Usage:
    python run_multi_paper.py                # Start paper trading
    python run_multi_paper.py --status       # Check status
    python run_multi_paper.py --stop         # Stop trading
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

# =============================================================================
# ASSET CONFIGURATIONS
# =============================================================================

ASSET_CONFIGS = {
    "TSLA": {
        "symbol": "TSLA",
        "asset_class": "stock",
        "model_dir": "data/models_v6_improved",
        "model_prefix": "TSLA_binary",
        "pred_horizon": 8,
        "confidence_threshold": 0.55,
        "position_size_pct": 0.30,        # 30% of total capital
        "stop_loss_pct": 0.015,           # 1.5% stop loss
        "take_profit_pct": 0.03,          # 3% take profit
        "check_interval_seconds": 60,     # 1 minute
        "data_source": "yfinance",
        "trading_hours": True,            # Only trade during market hours
    },
    "XRP_USDT": {
        "symbol": "XRP/USDT",
        "asset_class": "crypto",
        "model_dir": "data/models_v6_improved",
        "model_prefix": "XRP_USDT_binary",
        "pred_horizon": 3,
        "confidence_threshold": 0.58,
        "position_size_pct": 0.35,        # 35% of total capital
        "stop_loss_pct": 0.02,            # 2% stop loss
        "take_profit_pct": 0.04,          # 4% take profit
        "check_interval_seconds": 120,    # 2 minutes (crypto more volatile)
        "data_source": "ccxt",
        "trading_hours": False,           # 24/7 trading
    },
    "BTC_USDT": {
        "symbol": "BTC/USDT",
        "asset_class": "crypto",
        "model_dir": "data/models_v6_improved",
        "model_prefix": "BTC_USDT_binary",
        "pred_horizon": 3,
        "confidence_threshold": 0.60,
        "position_size_pct": 0.35,        # 35% of total capital
        "stop_loss_pct": 0.02,            # 2% stop loss
        "take_profit_pct": 0.04,          # 4% take profit
        "check_interval_seconds": 120,    # 2 minutes
        "data_source": "ccxt",
        "trading_hours": False,           # 24/7 trading
    },
}

PORTFOLIO_CONFIG = {
    "initial_capital": 10000,
    "max_total_exposure": 1.0,  # Max 100% invested at once
    "max_daily_loss_pct": 0.05,  # Stop trading if down 5% in a day
}


def setup_logging(log_dir: Path) -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"multi_paper_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    asset_class: str
    entry_price: float
    quantity: float
    entry_time: str
    stop_loss: float
    take_profit: float
    highest_price: float = 0.0  # For trailing stop
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "highest_price": self.highest_price,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Position":
        return cls(**d)


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    asset_class: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    entry_time: str
    exit_time: str
    
    def to_dict(self) -> Dict:
        return vars(self)


@dataclass
class AssetState:
    """State for a single asset."""
    symbol: str
    position: Optional[Position] = None
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    last_prediction: Optional[Dict] = None
    last_price: float = 0.0
    errors: int = 0


class MultiAssetPaperTrader:
    """
    Multi-asset paper trader using V6 improved models.
    """
    
    def __init__(
        self,
        asset_configs: Dict[str, Dict],
        portfolio_config: Dict,
    ):
        self.asset_configs = asset_configs
        self.portfolio_config = portfolio_config
        self.logger = logging.getLogger("MultiAssetTrader")
        self.running = False
        
        # State management
        self.state_file = Path("data/multi_paper/state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Portfolio state
        self.balance = portfolio_config["initial_capital"]
        self.starting_balance = portfolio_config["initial_capital"]
        self.daily_start_balance = self.balance
        
        # Asset states
        self.asset_states: Dict[str, AssetState] = {}
        for name in asset_configs:
            self.asset_states[name] = AssetState(symbol=name)
        
        # Models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.features: Dict[str, List[str]] = {}
        
        # Data fetchers
        self.ccxt_exchange = None
        
    def load_models(self) -> bool:
        """Load all asset models."""
        import joblib
        import importlib.util
        
        # Load feature extractor directly to avoid bot package import issues
        fe_path = Path("bot/ml/v6_feature_extractor.py")
        spec = importlib.util.spec_from_file_location("v6_feature_extractor", fe_path)
        v6_fe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v6_fe)
        self.feature_extractor = v6_fe.build_v6_features
        
        for name, config in self.asset_configs.items():
            model_dir = Path(config["model_dir"])
            prefix = config["model_prefix"]
            
            # Load model
            model_path = model_dir / f"{prefix}_ensemble_v6.pkl"
            if not model_path.exists():
                self.logger.error(f"Model not found: {model_path}")
                return False
            
            self.models[name] = joblib.load(model_path)
            self.logger.info(f"Loaded model for {name}")
            
            # Load scaler
            scaler_path = model_dir / f"{prefix}_scaler_v6.pkl"
            if scaler_path.exists():
                self.scalers[name] = joblib.load(scaler_path)
            
            # Load features
            features_path = model_dir / f"{prefix.replace('_binary', '')}_selected_features_v6.json"
            if features_path.exists():
                with open(features_path) as f:
                    self.features[name] = json.load(f)
            else:
                self.features[name] = None
        
        return True
    
    def init_data_sources(self) -> bool:
        """Initialize data sources."""
        try:
            import ccxt
            self.ccxt_exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            self.logger.info("Initialized CCXT Binance connection")
            return True
        except Exception as e:
            self.logger.error(f"Failed to init data sources: {e}")
            return False
    
    async def get_market_data(self, asset_name: str) -> Optional[Dict]:
        """Fetch current market data for an asset."""
        config = self.asset_configs[asset_name]
        
        try:
            if config["data_source"] == "yfinance":
                return await self._get_yfinance_data(config["symbol"])
            elif config["data_source"] == "ccxt":
                return await self._get_ccxt_data(config["symbol"])
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {asset_name}: {e}")
            self.asset_states[asset_name].errors += 1
            return None
    
    async def _get_yfinance_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from yfinance."""
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo", interval="1h")
        
        if df.empty or len(df) < 50:
            return None
        
        df.columns = [c.lower() for c in df.columns]
        
        return {
            "price": df["close"].iloc[-1],
            "data": df,
            "timestamp": datetime.now(timezone.utc),
        }
    
    async def _get_ccxt_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from CCXT."""
        loop = asyncio.get_event_loop()
        
        # Run sync CCXT in executor
        ohlcv = await loop.run_in_executor(
            None,
            lambda: self.ccxt_exchange.fetch_ohlcv(symbol, "1h", limit=500)
        )
        
        if not ohlcv or len(ohlcv) < 50:
            return None
        
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("datetime")
        df = df.drop("timestamp", axis=1)
        
        return {
            "price": df["close"].iloc[-1],
            "data": df,
            "timestamp": datetime.now(timezone.utc),
        }
    
    def generate_features(self, asset_name: str, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Generate features for prediction."""
        config = self.asset_configs[asset_name]
        
        try:
            features_df = self.feature_extractor(
                df, 
                pred_horizon=config["pred_horizon"],
                asset_class=config["asset_class"]
            )
            features_df = features_df.dropna()
            
            if features_df.empty:
                return None
            
            last_row = features_df.iloc[-1]
            
            # Select required features
            feature_list = self.features.get(asset_name)
            if feature_list:
                available = [f for f in feature_list if f in last_row.index]
                return {f: float(last_row[f]) for f in available if not pd.isna(last_row[f])}
            
            return last_row.to_dict()
            
        except Exception as e:
            self.logger.error(f"Feature generation failed for {asset_name}: {e}")
            return None
    
    def predict(self, asset_name: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate prediction from model."""
        try:
            model = self.models[asset_name]
            scaler = self.scalers.get(asset_name)
            feature_list = self.features.get(asset_name)
            
            # Prepare feature vector
            if feature_list:
                feature_names = [f for f in feature_list if f in features]
                X = np.array([[features[f] for f in feature_names]])
            else:
                X = np.array([list(features.values())])
            
            # Scale if available
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except:
                    pass
            
            # Predict
            pred = model.predict(X)[0]
            
            # Get probability
            confidence = 0.5
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
            
            return {
                "prediction": int(pred),
                "confidence": float(confidence),
                "signal": "BUY" if pred == 1 else "SELL",
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {asset_name}: {e}")
            return {"prediction": None, "confidence": 0.0, "signal": "HOLD"}
    
    def is_trading_allowed(self, asset_name: str) -> bool:
        """Check if trading is allowed for this asset right now."""
        config = self.asset_configs[asset_name]
        
        # Check daily loss limit
        daily_pnl = self.balance - self.daily_start_balance
        if daily_pnl < -self.portfolio_config["max_daily_loss_pct"] * self.starting_balance:
            self.logger.warning(f"Daily loss limit reached: ${daily_pnl:.2f}")
            return False
        
        # Check trading hours for stocks
        if config.get("trading_hours"):
            now = datetime.now()
            # Simple US market hours check (9:30 AM - 4:00 PM ET)
            # This is simplified - production should use proper market calendar
            if now.weekday() >= 5:  # Weekend
                return False
            hour = now.hour
            if hour < 14 or hour >= 21:  # Approximate UTC for US market hours
                return False
        
        return True
    
    def should_enter(self, asset_name: str, prediction: Dict, price: float) -> bool:
        """Determine if we should enter a position."""
        config = self.asset_configs[asset_name]
        state = self.asset_states[asset_name]
        
        # Already have position
        if state.position is not None:
            return False
        
        # Check confidence threshold
        if prediction["confidence"] < config["confidence_threshold"]:
            return False
        
        # Only enter on BUY signals
        if prediction["signal"] != "BUY":
            return False
        
        # Check total exposure
        total_invested = sum(
            s.position.quantity * s.last_price
            for s in self.asset_states.values()
            if s.position is not None
        )
        max_exposure = self.portfolio_config["max_total_exposure"] * self.balance
        
        position_value = self.balance * config["position_size_pct"]
        if total_invested + position_value > max_exposure:
            return False
        
        return True
    
    def should_exit(self, asset_name: str, price: float) -> Optional[str]:
        """Determine if we should exit a position."""
        config = self.asset_configs[asset_name]
        state = self.asset_states[asset_name]
        
        if state.position is None:
            return None
        
        pos = state.position
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        
        # Stop loss
        if pnl_pct < -config["stop_loss_pct"]:
            return "STOP_LOSS"
        
        # Take profit
        if pnl_pct > config["take_profit_pct"]:
            return "TAKE_PROFIT"
        
        # Signal reversal (if last prediction was strong SELL)
        if state.last_prediction:
            if (state.last_prediction["signal"] == "SELL" and 
                state.last_prediction["confidence"] > config["confidence_threshold"]):
                return "SIGNAL_REVERSAL"
        
        return None
    
    def execute_entry(self, asset_name: str, price: float) -> None:
        """Execute entry into a position."""
        config = self.asset_configs[asset_name]
        state = self.asset_states[asset_name]
        
        position_value = self.balance * config["position_size_pct"]
        quantity = position_value / price
        
        state.position = Position(
            symbol=asset_name,
            asset_class=config["asset_class"],
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=price * (1 - config["stop_loss_pct"]),
            take_profit=price * (1 + config["take_profit_pct"]),
            highest_price=price,
        )
        
        emoji = "📈" if config["asset_class"] == "stock" else "🪙"
        self.logger.info(
            f"{emoji} ENTERED {asset_name}: {quantity:.4f} @ ${price:.2f} "
            f"(SL: ${state.position.stop_loss:.2f}, TP: ${state.position.take_profit:.2f})"
        )
    
    def execute_exit(self, asset_name: str, price: float, reason: str) -> None:
        """Execute exit from a position."""
        config = self.asset_configs[asset_name]
        state = self.asset_states[asset_name]
        pos = state.position
        
        pnl = (price - pos.entry_price) * pos.quantity
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        
        # Update balances
        self.balance += pnl
        state.total_pnl += pnl
        
        # Record trade
        trade = Trade(
            symbol=asset_name,
            asset_class=config["asset_class"],
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc).isoformat(),
        )
        state.trades.append(trade)
        
        # Clear position
        state.position = None
        
        emoji = "✅" if pnl > 0 else "❌"
        self.logger.info(
            f"{emoji} EXITED {asset_name} [{reason}]: ${pnl:.2f} ({pnl_pct:.2f}%) | "
            f"Balance: ${self.balance:.2f}"
        )
    
    def save_state(self) -> None:
        """Save current state to file."""
        # Prepare asset states
        assets = {}
        for name, state in self.asset_states.items():
            assets[name] = {
                "position": state.position.to_dict() if state.position else None,
                "trades": [t.to_dict() for t in state.trades[-20:]],  # Last 20 trades
                "total_pnl": state.total_pnl,
                "errors": state.errors,
            }
        
        # Calculate stats
        all_trades = []
        for state in self.asset_states.values():
            all_trades.extend(state.trades)
        
        winning_trades = sum(1 for t in all_trades if t.pnl > 0)
        
        state = {
            "balance": self.balance,
            "starting_balance": self.starting_balance,
            "daily_start_balance": self.daily_start_balance,
            "total_pnl": self.balance - self.starting_balance,
            "total_trades": len(all_trades),
            "winning_trades": winning_trades,
            "assets": assets,
            "last_update": datetime.now(timezone.utc).isoformat(),
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
            
            self.balance = state.get("balance", self.starting_balance)
            self.daily_start_balance = state.get("daily_start_balance", self.balance)
            
            # Check if it's a new day
            last_update = state.get("last_update", "")
            if last_update:
                last_date = datetime.fromisoformat(last_update.replace("Z", "+00:00")).date()
                if last_date < datetime.now(timezone.utc).date():
                    self.daily_start_balance = self.balance
            
            # Load asset states
            for name, asset_data in state.get("assets", {}).items():
                if name in self.asset_states:
                    if asset_data.get("position"):
                        self.asset_states[name].position = Position.from_dict(asset_data["position"])
                    self.asset_states[name].total_pnl = asset_data.get("total_pnl", 0)
            
            self.logger.info(f"Loaded state: Balance=${self.balance:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    async def process_asset(self, asset_name: str) -> None:
        """Process a single asset - check for signals and manage position."""
        config = self.asset_configs[asset_name]
        state = self.asset_states[asset_name]
        
        # Check if trading is allowed
        if not self.is_trading_allowed(asset_name):
            self.logger.debug(f"{asset_name}: Trading not allowed (market hours/daily limit)")
            return
        
        # Get market data
        self.logger.debug(f"{asset_name}: Fetching market data...")
        market = await self.get_market_data(asset_name)
        if not market:
            self.logger.warning(f"{asset_name}: No market data available")
            return
        
        price = market["price"]
        state.last_price = price
        self.logger.info(f"{asset_name}: Price = ${price:.2f}")
        
        # Generate features
        features = self.generate_features(asset_name, market["data"])
        if not features:
            self.logger.warning(f"{asset_name}: No features generated")
            return
        
        # Get prediction
        prediction = self.predict(asset_name, features)
        state.last_prediction = prediction
        
        self.logger.info(
            f"{asset_name}: Signal={prediction['signal']} "
            f"Confidence={prediction['confidence']:.1%}"
        )
        
        # Check for exit first (if we have a position)
        if state.position:
            exit_reason = self.should_exit(asset_name, price)
            if exit_reason:
                self.execute_exit(asset_name, price, exit_reason)
                self.save_state()
                return
        
        # Check for entry
        if self.should_enter(asset_name, prediction, price):
            self.execute_entry(asset_name, price)
            self.save_state()
    
    async def trading_loop(self) -> None:
        """Main trading loop for all assets."""
        iteration = 0
        last_check = {name: datetime.now() - timedelta(seconds=300) for name in self.asset_configs}  # Start immediately
        
        self.logger.info("Starting trading loop...")
        
        while self.running:
            iteration += 1
            now = datetime.now()
            
            # Process each asset based on its check interval
            for asset_name, config in self.asset_configs.items():
                interval = config["check_interval_seconds"]
                elapsed = (now - last_check[asset_name]).total_seconds()
                if elapsed >= interval:
                    self.logger.info(f"Processing {asset_name}...")
                    try:
                        await self.process_asset(asset_name)
                    except Exception as e:
                        self.logger.error(f"Error processing {asset_name}: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                    last_check[asset_name] = now
            
            # Log portfolio status periodically
            if iteration % 30 == 0:
                self.log_portfolio_status()
            
            # Short sleep to prevent CPU spinning
            await asyncio.sleep(5)
    
    def log_portfolio_status(self) -> None:
        """Log current portfolio status."""
        total_pnl = self.balance - self.starting_balance
        total_pnl_pct = (total_pnl / self.starting_balance) * 100
        
        positions = []
        for name, state in self.asset_states.items():
            if state.position:
                pos = state.position
                pnl_pct = (state.last_price - pos.entry_price) / pos.entry_price * 100
                positions.append(f"{name}:{pnl_pct:+.1f}%")
        
        pos_str = ", ".join(positions) if positions else "FLAT"
        
        self.logger.info(
            f"📊 PORTFOLIO | Balance: ${self.balance:.2f} | "
            f"P&L: ${total_pnl:.2f} ({total_pnl_pct:+.2f}%) | "
            f"Positions: {pos_str}"
        )
    
    async def start(self) -> None:
        """Start paper trading."""
        self.logger.info("=" * 70)
        self.logger.info("MULTI-ASSET PAPER TRADING - V6 IMPROVED MODELS")
        self.logger.info("=" * 70)
        
        # Print asset configs
        for name, config in self.asset_configs.items():
            self.logger.info(
                f"  {name}: {config['asset_class']} | "
                f"Position: {config['position_size_pct']*100:.0f}% | "
                f"Confidence: {config['confidence_threshold']*100:.0f}%"
            )
        
        self.logger.info(f"Initial Capital: ${self.portfolio_config['initial_capital']:.2f}")
        self.logger.info("=" * 70)
        
        # Load models
        if not self.load_models():
            self.logger.error("Failed to load models, exiting")
            return
        
        # Init data sources
        if not self.init_data_sources():
            self.logger.error("Failed to initialize data sources")
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
        all_trades = []
        for state in self.asset_states.values():
            all_trades.extend(state.trades)
        
        win_rate = 0.0
        if all_trades:
            wins = sum(1 for t in all_trades if t.pnl > 0)
            win_rate = wins / len(all_trades) * 100
        
        asset_status = {}
        for name, state in self.asset_states.items():
            asset_trades = state.trades
            asset_wins = sum(1 for t in asset_trades if t.pnl > 0)
            asset_status[name] = {
                "position": state.position.to_dict() if state.position else None,
                "last_price": state.last_price,
                "total_pnl": state.total_pnl,
                "trades": len(asset_trades),
                "win_rate": (asset_wins / len(asset_trades) * 100) if asset_trades else 0,
            }
        
        return {
            "running": self.running,
            "balance": self.balance,
            "starting_balance": self.starting_balance,
            "total_pnl": self.balance - self.starting_balance,
            "total_pnl_pct": ((self.balance - self.starting_balance) / self.starting_balance) * 100,
            "total_trades": len(all_trades),
            "win_rate": win_rate,
            "assets": asset_status,
        }


async def run_paper_trading(args) -> None:
    """Run paper trading."""
    log_dir = Path("data/multi_paper/logs")
    setup_logging(log_dir)
    
    # Update capital if provided
    portfolio_config = PORTFOLIO_CONFIG.copy()
    portfolio_config["initial_capital"] = args.capital
    
    trader = MultiAssetPaperTrader(ASSET_CONFIGS, portfolio_config)
    
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
    logging.info("\n" + "=" * 70)
    logging.info("FINAL STATUS")
    logging.info(f"Balance: ${status['balance']:.2f}")
    logging.info(f"P&L: ${status['total_pnl']:.2f} ({status['total_pnl_pct']:.2f}%)")
    logging.info(f"Trades: {status['total_trades']} (Win Rate: {status['win_rate']:.1f}%)")
    logging.info("=" * 70)


def show_status() -> None:
    """Show current status."""
    state_file = Path("data/multi_paper/state.json")
    
    if not state_file.exists():
        print("No multi-asset paper trading state found. Start trading first.")
        return
    
    with open(state_file) as f:
        state = json.load(f)
    
    print("\n" + "=" * 70)
    print("MULTI-ASSET PAPER TRADING STATUS")
    print("=" * 70)
    
    print(f"\n💰 PORTFOLIO")
    print(f"   Balance: ${state['balance']:.2f}")
    print(f"   Starting: ${state['starting_balance']:.2f}")
    print(f"   Total P&L: ${state['total_pnl']:.2f}")
    pnl_pct = (state['total_pnl'] / state['starting_balance']) * 100
    print(f"   Return: {pnl_pct:+.2f}%")
    print(f"   Trades: {state['total_trades']} (Wins: {state['winning_trades']})")
    
    if state['total_trades'] > 0:
        print(f"   Win Rate: {state['winning_trades']/state['total_trades']*100:.1f}%")
    
    print(f"\n📊 ASSETS")
    for name, asset in state.get("assets", {}).items():
        emoji = "📈" if ASSET_CONFIGS[name]["asset_class"] == "stock" else "🪙"
        print(f"\n   {emoji} {name}")
        print(f"      P&L: ${asset['total_pnl']:.2f}")
        print(f"      Trades: {len(asset.get('trades', []))}")
        
        if asset.get("position"):
            pos = asset["position"]
            print(f"      📍 OPEN: {pos['quantity']:.4f} @ ${pos['entry_price']:.2f}")
            print(f"         SL: ${pos['stop_loss']:.2f} | TP: ${pos['take_profit']:.2f}")
        else:
            print(f"      No open position")
    
    print(f"\n⏰ Last Update: {state.get('last_update', 'unknown')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Paper Trading - V6 Models")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    else:
        asyncio.run(run_paper_trading(args))


if __name__ == "__main__":
    main()
