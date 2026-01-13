#!/usr/bin/env python3
"""
Enhanced ML Paper Trading with AI Features.

Integrates all AI enhancements:
- Auto-retraining scheduler (monitors model health, triggers retraining)
- Online learning (learns from every trade outcome)
- LLM advisor (market insights via local Ollama)
- Streaming features (incremental feature calculation)

Usage:
    python run_ml_paper_trading_enhanced.py
    python run_ml_paper_trading_enhanced.py --capital 10000 --interval 60
    python run_ml_paper_trading_enhanced.py --no-llm  # Disable LLM features
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
LOG_DIR = Path("data/ml_paper_trading_enhanced/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "enhanced_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("enhanced-ml-trading")

# Reduce noise from other loggers
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("bot.ml.auto_retrainer").setLevel(logging.WARNING)

try:
    import yfinance as yf
except ImportError:
    logger.error("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Core ML imports
from bot.ml.predictor import MLPredictor, PredictionResult
from bot.ml.feature_engineer import FeatureEngineer

# AI Enhancement imports
from bot.ml import (
    AutoRetrainingScheduler,
    ModelHealth,
    ModelHealthStatus,
    PerformanceMetrics,
    OnlineLearningManager,
    ExperienceBuffer,
    StreamingFeatureEngineer,
)

# LLM imports
try:
    from bot.llm import LLMAdvisor
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    LLMAdvisor = None


SYMBOL_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
}


class Position:
    """Trading position."""
    def __init__(self, symbol: str, quantity: float, entry_price: float, side: str,
                 features: Optional[np.ndarray] = None, confidence: float = 0.5):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.side = side  # "LONG" or "SHORT"
        self.entry_time = datetime.now()
        self.entry_features = features  # Store features for online learning
        self.entry_confidence = confidence

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
        }


class EnhancedMLTrader:
    """
    Enhanced ML Paper Trader with AI features.

    Features:
    - Aggressive trading (always takes a position)
    - Auto-retraining when model performance degrades
    - Online learning from trade outcomes
    - LLM-powered market insights
    - Streaming feature calculation
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 0.3,
        model_type: str = "gradient_boosting",
        data_dir: str = "data/ml_paper_trading_enhanced",
        enable_auto_retraining: bool = True,
        enable_online_learning: bool = True,
        enable_llm: bool = True,
        llm_model: str = "mistral",
        retraining_check_hours: float = 6,
        online_learning_frequency: int = 10,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size = position_size
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Feature flags
        self.enable_auto_retraining = enable_auto_retraining
        self.enable_online_learning = enable_online_learning
        self.enable_llm = enable_llm and HAS_LLM

        # Core components
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.models: Dict[str, MLPredictor] = {}

        # Price cache
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, datetime] = {}

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Streaming feature engineers per symbol
        self.streaming_features: Dict[str, StreamingFeatureEngineer] = {}

        # Load models
        self._load_models()

        # Initialize AI components
        self._init_auto_retraining(retraining_check_hours)
        self._init_online_learning(online_learning_frequency)
        self._init_llm_advisor(llm_model)

        # Performance tracking
        self._daily_returns: List[float] = []
        self._win_count = 0
        self._loss_count = 0
        self._last_health_check = datetime.now()

        # LLM insights cache
        self._llm_insights: Dict[str, Any] = {}
        self._last_llm_analysis = datetime.now() - timedelta(hours=1)

        logger.info(f"Enhanced ML Trader initialized")
        logger.info(f"  Auto-retraining: {'ON' if self.enable_auto_retraining else 'OFF'}")
        logger.info(f"  Online learning: {'ON' if self.enable_online_learning else 'OFF'}")
        logger.info(f"  LLM advisor: {'ON' if self.enable_llm else 'OFF'}")

    def _load_models(self):
        """Load ML models for each symbol."""
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

                    # Initialize streaming feature engineer
                    self.streaming_features[symbol] = StreamingFeatureEngineer()
                else:
                    logger.warning(f"Could not load model for {symbol}")
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")

        if not self.models:
            logger.error("No models loaded! Run training first.")
            sys.exit(1)

    def _init_auto_retraining(self, check_hours: float):
        """Initialize auto-retraining scheduler."""
        if not self.enable_auto_retraining:
            self.auto_retrainer = None
            return

        self.auto_retrainer = AutoRetrainingScheduler(
            check_interval_hours=check_hours,
            data_dir=str(self.data_dir / "auto_retraining"),
        )

        # Register models
        for symbol in self.models:
            self.auto_retrainer.register_model(
                symbol=symbol,
                model_type=self.model_type,
                model=self.models[symbol],
            )

        logger.info(f"Auto-retraining scheduler initialized (check every {check_hours}h)")

    def _init_online_learning(self, update_frequency: int):
        """Initialize online learning manager."""
        if not self.enable_online_learning:
            self.online_learners: Dict[str, OnlineLearningManager] = {}
            return

        self.online_learners: Dict[str, OnlineLearningManager] = {}
        for symbol, model in self.models.items():
            self.online_learners[symbol] = OnlineLearningManager(
                model=model.model if hasattr(model, 'model') else None,
                buffer_size=1000,
                update_frequency=update_frequency,
                data_dir=str(self.data_dir / "online_learning" / symbol.replace("/", "_")),
            )

        logger.info(f"Online learning initialized (update every {update_frequency} trades)")

    def _init_llm_advisor(self, llm_model: str):
        """Initialize LLM advisor."""
        if not self.enable_llm:
            self.llm_advisor = None
            return

        try:
            self.llm_advisor = LLMAdvisor(model=llm_model)
            if self.llm_advisor.is_available():
                logger.info(f"LLM advisor initialized (model: {llm_model})")
            else:
                logger.warning("LLM advisor not available (Ollama not running?)")
                self.llm_advisor = None
                self.enable_llm = False
        except Exception as e:
            logger.warning(f"Failed to initialize LLM advisor: {e}")
            self.llm_advisor = None
            self.enable_llm = False

    def fetch_prices(self, symbol: str, period: str = "60d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch price data from Yahoo Finance."""
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

            # Update streaming features
            if symbol in self.streaming_features:
                latest = result.iloc[-1]
                self.streaming_features[symbol].update(
                    open_=latest["open"],
                    high=latest["high"],
                    low=latest["low"],
                    close=latest["close"],
                    volume=latest["volume"],
                )

            return result

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return self._price_cache.get(symbol)

    def get_aggressive_signal(self, symbol: str) -> Optional[Dict]:
        """Get aggressive signal with streaming features."""
        if symbol not in self.models:
            return None

        df = self.fetch_prices(symbol)
        if df is None or len(df) < 100:
            return None

        try:
            # Get prediction
            prediction = self.models[symbol].predict(df)

            # Force LONG or SHORT based on probabilities
            prob_long = prediction.probability_long
            prob_short = prediction.probability_short

            if prob_long > prob_short:
                action = "LONG"
                confidence = prob_long / (prob_long + prob_short)
            else:
                action = "SHORT"
                confidence = prob_short / (prob_long + prob_short)

            # Get streaming features for online learning
            streaming_feats = None
            if symbol in self.streaming_features:
                streaming_feats = self.streaming_features[symbol]._calculate_features()

            return {
                "action": action,
                "confidence": confidence,
                "prob_long": prob_long,
                "prob_short": prob_short,
                "prob_flat": prediction.probability_flat,
                "original_action": prediction.action,
                "features": streaming_feats,
            }
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
            if pos.side == "LONG":
                total += pos.quantity * current_price
            else:  # SHORT
                pnl = (pos.entry_price - current_price) * pos.quantity
                total += pos.quantity * pos.entry_price + pnl
        return total

    def execute_trade(self, symbol: str, action: str, price: float,
                      features: Optional[np.ndarray] = None,
                      confidence: float = 0.5) -> Optional[Dict]:
        """Execute a trade with online learning integration."""
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

                self.positions[symbol] = Position(
                    symbol, quantity, price, action,
                    features=features, confidence=confidence
                )
                trade["quantity"] = quantity
                trade["value"] = trade_value
                trade["side"] = action
                trade["status"] = "executed"

        self.trade_history.append(trade)
        return trade

    def _close_position(self, symbol: str, price: float) -> Optional[Dict]:
        """Close position and record for online learning."""
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
        pnl_pct = pnl / (pos.quantity * pos.entry_price)

        # Record outcome for online learning
        if self.enable_online_learning and symbol in self.online_learners:
            if pos.entry_features is not None:
                hold_time = (datetime.now() - pos.entry_time).total_seconds() / 60
                self.online_learners[symbol].record_trade_outcome(
                    symbol=symbol,
                    features=pos.entry_features,
                    action=pos.side,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    hold_time_minutes=int(hold_time),
                    confidence=pos.entry_confidence,
                    regime="unknown",
                )

        # Track wins/losses
        if pnl > 0:
            self._win_count += 1
        elif pnl < 0:
            self._loss_count += 1

        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": f"CLOSE_{pos.side}",
            "price": price,
            "quantity": pos.quantity,
            "value": value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_time_minutes": (datetime.now() - pos.entry_time).total_seconds() / 60,
            "status": "executed",
        }

        del self.positions[symbol]
        return trade

    def check_model_health(self) -> Dict[str, Any]:
        """Check health of all models."""
        if not self.enable_auto_retraining or not self.auto_retrainer:
            return {}

        health_report = {}
        for symbol in self.models:
            health = self.auto_retrainer.check_model_health(symbol, self.model_type)
            health_report[symbol] = {
                "status": health.status.value,
                "needs_retraining": health.needs_retraining,
                "reason": health.reason,
            }

            if health.needs_retraining:
                logger.warning(f"⚠️  {symbol} model needs retraining: {health.reason}")

        return health_report

    def record_performance_metrics(self):
        """Record performance metrics for auto-retraining monitoring."""
        if not self.enable_auto_retraining or not self.auto_retrainer:
            return

        # Calculate metrics from recent trades
        recent_trades = [t for t in self.trade_history[-50:] if "pnl" in t]
        if len(recent_trades) < 5:
            return

        returns = [t["pnl_pct"] for t in recent_trades if "pnl_pct" in t]
        wins = sum(1 for r in returns if r > 0)

        accuracy = wins / len(returns) if returns else 0.5
        win_rate = wins / len(returns) if returns else 0.5
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if returns else 0

        # Record for each symbol
        for symbol in self.models:
            metrics = PerformanceMetrics(
                accuracy=accuracy,
                precision=accuracy,
                recall=accuracy,
                f1_score=accuracy,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                profit_factor=1.0 + np.mean(returns) if returns else 1.0,
                max_drawdown=0.0,
            )
            self.auto_retrainer.record_performance(symbol, self.model_type, metrics)

    def get_llm_market_insight(self, symbol: str) -> Optional[str]:
        """Get LLM-powered market insight."""
        if not self.enable_llm or not self.llm_advisor:
            return None

        # Rate limit LLM calls (once per 30 minutes)
        if (datetime.now() - self._last_llm_analysis).total_seconds() < 1800:
            return self._llm_insights.get(symbol)

        try:
            df = self.fetch_prices(symbol)
            if df is None:
                return None

            # Calculate metrics for context
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365) * 100
            trend = "bullish" if df["close"].iloc[-1] > df["close"].iloc[-20] else "bearish"

            context = {
                "symbol": symbol,
                "current_price": float(df["close"].iloc[-1]),
                "24h_change": float(returns.iloc[-24:].sum() * 100) if len(returns) >= 24 else 0,
                "volatility": float(volatility),
                "trend": trend,
            }

            insight = self.llm_advisor.analyze_market(context)
            self._llm_insights[symbol] = insight
            self._last_llm_analysis = datetime.now()

            return insight
        except Exception as e:
            logger.debug(f"LLM insight failed: {e}")
            return None

    def run_iteration(self) -> Dict[str, Any]:
        """Run one trading iteration with all AI features."""
        iteration_result = {
            "timestamp": datetime.now().isoformat(),
            "signals": {},
            "trades": [],
            "portfolio_value": 0,
            "cash": self.cash,
            "positions": {},
            "ai_features": {},
        }

        # Periodic model health check
        if (datetime.now() - self._last_health_check).total_seconds() > 3600:
            health = self.check_model_health()
            if health:
                iteration_result["ai_features"]["model_health"] = health
            self._last_health_check = datetime.now()

            # Record performance metrics
            self.record_performance_metrics()

        # Process each symbol
        for symbol in SYMBOL_MAP.keys():
            signal = self.get_aggressive_signal(symbol)
            if signal is None:
                continue

            current_price = self.get_current_price(symbol)
            current_position = self.positions.get(symbol)

            signal_info = {
                "action": signal["action"],
                "confidence": signal["confidence"],
                "prob_long": signal["prob_long"],
                "prob_short": signal["prob_short"],
                "price": current_price,
            }
            iteration_result["signals"][symbol] = signal_info

            # Log signal
            signal_emoji = "🟢" if signal["action"] == "LONG" else "🔴"
            pos_info = f"{current_position.side}" if current_position else "None"

            logger.info(
                f"  {symbol}: ${current_price:,.2f} | {signal_emoji} {signal['action']} "
                f"({signal['confidence']:.1%}) | Pos: {pos_info} | "
                f"[L:{signal['prob_long']:.0%} S:{signal['prob_short']:.0%}]"
            )

            # Get LLM insight (rate-limited)
            if self.enable_llm:
                insight = self.get_llm_market_insight(symbol)
                if insight:
                    iteration_result["ai_features"].setdefault("llm_insights", {})[symbol] = insight[:200]

            # Trading logic
            if current_position is None:
                trade = self.execute_trade(
                    symbol, signal["action"], current_price,
                    features=signal.get("features"),
                    confidence=signal["confidence"],
                )
                if trade and trade["status"] == "executed":
                    logger.info(f"    ➡️  OPEN {signal['action']} {trade['quantity']:.6f} @ ${current_price:,.2f}")
                    iteration_result["trades"].append(trade)
            elif current_position.side != signal["action"]:
                trade = self.execute_trade(
                    symbol, signal["action"], current_price,
                    features=signal.get("features"),
                    confidence=signal["confidence"],
                )
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

        # Store latest signals for dashboard
        self.latest_signals = iteration_result["signals"]

        # Add online learning stats
        if self.enable_online_learning:
            ol_stats = {}
            for symbol, learner in self.online_learners.items():
                status = learner.get_status()
                ol_stats[symbol] = {
                    "trade_count": status["trade_count"],
                    "update_count": status["update_count"],
                    "buffer_size": status["buffer"]["size"],
                }
            iteration_result["ai_features"]["online_learning"] = ol_stats

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
            "mode": "enhanced_ai",
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "position_count": len(self.positions),
            "trade_count": len(self.trade_history),
            "signals": {
                sym: {
                    "signal": sig.get("action"),
                    "regime": sig.get("regime", "unknown"),
                    "confidence": sig.get("confidence", 0),
                    "price": sig.get("price", 0),
                }
                for sym, sig in getattr(self, "latest_signals", {}).items()
            },
            "ai_features": {
                "auto_retraining_enabled": self.enable_auto_retraining,
                "online_learning_enabled": self.enable_online_learning,
                "llm_enabled": self.enable_llm,
                "win_count": self._win_count,
                "loss_count": self._loss_count,
                "win_rate": self._win_count / max(1, self._win_count + self._loss_count),
            },
        }

        with open(self.data_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        with open(self.data_dir / "equity.json", "w") as f:
            json.dump(self.equity_history[-1000:], f, indent=2)

        with open(self.data_dir / "trades.json", "w") as f:
            json.dump(self.trade_history[-100:], f, indent=2)

    def get_summary(self) -> Dict:
        """Get trading session summary."""
        total_pnl = self.portfolio_value - self.initial_capital
        trades_with_pnl = [t for t in self.trade_history if "pnl" in t]
        winning_trades = [t for t in trades_with_pnl if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades_with_pnl if t.get("pnl", 0) < 0]

        summary = {
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
            "ai_features": {
                "auto_retraining": self.enable_auto_retraining,
                "online_learning": self.enable_online_learning,
                "llm_advisor": self.enable_llm,
            },
        }

        # Add online learning stats
        if self.enable_online_learning:
            total_buffer = sum(len(l.buffer) for l in self.online_learners.values())
            total_updates = sum(l.update_count for l in self.online_learners.values())
            summary["online_learning_stats"] = {
                "total_experiences": total_buffer,
                "model_updates": total_updates,
            }

        return summary

    def shutdown(self):
        """Clean shutdown of AI components."""
        logger.info("Shutting down AI components...")

        if self.enable_auto_retraining and self.auto_retrainer:
            self.auto_retrainer.stop()
            logger.info("  Auto-retraining scheduler stopped")

        self.save_state()
        logger.info("State saved")


def run_enhanced_trading(
    initial_capital: float = 10000.0,
    interval: int = 60,
    model_type: str = "gradient_boosting",
    max_iterations: int = 0,
    enable_llm: bool = True,
    enable_auto_retraining: bool = True,
    enable_online_learning: bool = True,
):
    """Run enhanced ML paper trading with AI features."""
    logger.info("=" * 70)
    logger.info("🤖 ENHANCED ML PAPER TRADING WITH AI FEATURES")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Mode: AGGRESSIVE + AI ENHANCED")
    logger.info(f"Loop Interval: {interval}s")
    logger.info("-" * 70)
    logger.info("AI Features:")
    logger.info(f"  🔄 Auto-retraining: {'ENABLED' if enable_auto_retraining else 'DISABLED'}")
    logger.info(f"  📚 Online learning: {'ENABLED' if enable_online_learning else 'DISABLED'}")
    logger.info(f"  🧠 LLM advisor: {'ENABLED' if enable_llm else 'DISABLED'}")
    logger.info("=" * 70)

    trader = EnhancedMLTrader(
        initial_capital=initial_capital,
        model_type=model_type,
        enable_auto_retraining=enable_auto_retraining,
        enable_online_learning=enable_online_learning,
        enable_llm=enable_llm,
    )

    logger.info(f"Loaded {len(trader.models)} models")
    logger.info("Starting enhanced trading... (Ctrl+C to stop)")
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

            # Show AI feature activity
            if result.get("ai_features"):
                if "model_health" in result["ai_features"]:
                    logger.info("  🔍 Model health checked")
                if "online_learning" in result["ai_features"]:
                    ol = result["ai_features"]["online_learning"]
                    total_trades = sum(s["trade_count"] for s in ol.values())
                    total_updates = sum(s["update_count"] for s in ol.values())
                    if total_trades > 0:
                        logger.info(f"  📚 Online learning: {total_trades} trades, {total_updates} updates")

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

    # Clean shutdown
    trader.shutdown()

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("🤖 ENHANCED TRADING SESSION ENDED")
    logger.info("=" * 70)

    summary = trader.get_summary()
    pnl_emoji = "📈" if summary["total_pnl"] >= 0 else "📉"

    logger.info(f"Final Portfolio: ${summary['portfolio_value']:,.2f}")
    logger.info(f"Total P&L: {pnl_emoji} ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.2f}%)")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Closed Trades: {summary['closed_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']:.1f}%")
    logger.info(f"Open Positions: {summary['positions']}")

    if "online_learning_stats" in summary:
        ol = summary["online_learning_stats"]
        logger.info(f"Online Learning: {ol['total_experiences']} experiences, {ol['model_updates']} updates")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Enhanced ML Paper Trading with AI Features")
    parser.add_argument("--capital", "-c", type=float, default=10000.0)
    parser.add_argument("--interval", "-i", type=int, default=60)
    parser.add_argument("--model", "-m", choices=["random_forest", "gradient_boosting"], default="gradient_boosting")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM advisor")
    parser.add_argument("--no-retraining", action="store_true", help="Disable auto-retraining")
    parser.add_argument("--no-online-learning", action="store_true", help="Disable online learning")

    args = parser.parse_args()

    run_enhanced_trading(
        initial_capital=args.capital,
        interval=args.interval,
        model_type=args.model,
        max_iterations=args.max_iterations,
        enable_llm=not args.no_llm,
        enable_auto_retraining=not args.no_retraining,
        enable_online_learning=not args.no_online_learning,
    )


if __name__ == "__main__":
    main()
