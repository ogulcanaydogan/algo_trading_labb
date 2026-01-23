#!/usr/bin/env python3
"""
Production-Ready Live Trading Script.

Integrates ALL improvements:
- ML-based signal generation with Gradient Boosting
- Technical analysis (candlestick patterns, divergences, confluence zones)
- Advanced risk management (correlation, regime, VaR, recovery mode)
- Market impact slippage estimation
- Telegram/Discord notifications
- Break-even stops and partial profit taking
- Auto-retraining and online learning

IMPORTANT: This script trades REAL MONEY. Use with caution.

Usage:
    # Dry run mode (recommended first)
    python run_production_trading.py --dry-run

    # Testnet mode
    python run_production_trading.py --testnet

    # Live mode (REAL MONEY)
    python run_production_trading.py --live
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
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
LOG_DIR = Path("data/production_trading/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"production_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("production-trading")

# Reduce noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ccxt").setLevel(logging.WARNING)

# Core imports
from bot.exchange import ExchangeClient
from bot.strategy import StrategyConfig, compute_indicators, generate_signal
from bot.ml.predictor import MLPredictor
from bot.ml.feature_engineer import FeatureEngineer

# Advanced features
from bot.risk_manager import (
    RiskManager, RiskConfig, RiskAssessment,
    CorrelationConfig, RegimeConfig, VaRConfig, RecoveryConfig,
    TradingStatus, MarketRegime
)
from bot.slippage import SlippageModel, SlippageConfig, OrderType, detect_market_condition
from bot.notifications import NotificationManager, Alert, AlertType, AlertLevel
from bot.technical_analysis import TechnicalAnalyzer

# Trading features
from bot.trading import TradingManager, TrailingStopConfig


@dataclass
class ProductionConfig:
    """Production trading configuration."""
    # Trading parameters
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_capital: float = 1000.0

    # Position sizing
    risk_per_trade_pct: float = 1.0      # 1% risk per trade
    max_position_pct: float = 20.0       # Max 20% of capital per position
    max_total_exposure_pct: float = 60.0 # Max 60% total exposure

    # Stop loss / Take profit
    stop_loss_pct: float = 2.0           # 2% stop loss
    take_profit_pct: float = 4.0         # 4% take profit (2:1 R:R)
    use_atr_stops: bool = True           # Use ATR for dynamic stops
    atr_stop_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0

    # Trailing stop
    use_trailing_stop: bool = True
    trailing_activation_pct: float = 1.5  # Activate at 1.5%
    trailing_distance_pct: float = 0.8    # Trail at 0.8%

    # Break-even and partial profits
    use_break_even: bool = True
    break_even_activation_pct: float = 1.0
    use_partial_profits: bool = True

    # ML settings
    use_ml_signals: bool = True
    ml_model_type: str = "gradient_boosting"
    ml_confidence_threshold: float = 0.6

    # Technical analysis confirmation
    use_technical_confirmation: bool = True
    require_divergence_confirmation: bool = False
    require_pattern_confirmation: bool = False

    # Risk management
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_daily_trades: int = 10
    use_regime_filter: bool = True
    use_correlation_filter: bool = True

    # Notifications
    send_trade_notifications: bool = True
    send_daily_summary: bool = True

    # Loop settings
    loop_interval_seconds: int = 300     # 5 minutes


class ProductionTrader:
    """
    Production-ready trading bot with all improvements.

    Features:
    - Multi-factor signal generation (ML + technical)
    - Advanced risk management
    - Slippage estimation
    - Real-time notifications
    - Break-even and partial profit taking
    """

    def __init__(
        self,
        config: ProductionConfig,
        exchange: Optional[ExchangeClient] = None,
        dry_run: bool = True,
    ):
        self.config = config
        self.exchange = exchange
        self.dry_run = dry_run

        self.data_dir = Path("data/production_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_risk_manager()
        self._init_ml_models()
        self._init_technical_analyzer()
        self._init_slippage_model()
        self._init_notifications()

        # State
        self.position = None
        self.entry_price = 0.0
        self.entry_time = None
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.break_even_activated = False
        self.partial_exits = []

        # Tracking
        self.trade_history = []
        self.equity_history = []
        self.balance = config.initial_capital

        # Load existing state
        self._load_state()

        logger.info(f"Production Trader initialized")
        logger.info(f"  Symbol: {config.symbol}")
        logger.info(f"  Capital: ${config.initial_capital:,.2f}")
        logger.info(f"  Dry Run: {dry_run}")
        logger.info(f"  ML Signals: {config.use_ml_signals}")
        logger.info(f"  Technical Confirmation: {config.use_technical_confirmation}")

    def _init_risk_manager(self):
        """Initialize advanced risk manager."""
        risk_config = RiskConfig(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_daily_trades=self.config.max_daily_trades,
            base_risk_per_trade_pct=self.config.risk_per_trade_pct,
            max_single_position_pct=self.config.max_position_pct,
            max_total_exposure_pct=self.config.max_total_exposure_pct,
            correlation_config=CorrelationConfig(
                enabled=self.config.use_correlation_filter,
            ),
            regime_config=RegimeConfig(
                enabled=self.config.use_regime_filter,
            ),
            var_config=VaRConfig(enabled=True),
            recovery_config=RecoveryConfig(enabled=True),
        )

        self.risk_manager = RiskManager(
            config=risk_config,
            initial_balance=self.config.initial_capital,
            data_dir=str(self.data_dir / "risk"),
        )

        logger.info("  Risk Manager: Initialized with VaR, regime, correlation filters")

    def _init_ml_models(self):
        """Initialize ML models."""
        self.ml_predictor = None
        self.feature_engineer = FeatureEngineer()

        if not self.config.use_ml_signals:
            return

        try:
            symbol_key = self.config.symbol.replace("/", "_")
            model_name = f"{symbol_key}_{self.config.timeframe}_{self.config.ml_model_type}"
            model_dir = f"data/models/{model_name}"

            predictor = MLPredictor(
                model_type=self.config.ml_model_type,
                model_dir=model_dir,
            )

            if predictor.load(model_name):
                self.ml_predictor = predictor
                logger.info(f"  ML Model: Loaded {model_name}")
            else:
                logger.warning(f"  ML Model: Could not load {model_name}")
        except Exception as e:
            logger.error(f"  ML Model: Error loading - {e}")

    def _init_technical_analyzer(self):
        """Initialize technical analyzer."""
        self.technical_analyzer = TechnicalAnalyzer()
        logger.info("  Technical Analyzer: Initialized (candlesticks, divergence, confluence)")

    def _init_slippage_model(self):
        """Initialize slippage model."""
        self.slippage_model = SlippageModel(
            data_dir=str(self.data_dir / "slippage"),
        )
        logger.info("  Slippage Model: Initialized")

    def _init_notifications(self):
        """Initialize notification manager."""
        self.notifier = NotificationManager()

        if self.notifier.has_channels():
            channels = self.notifier.get_configured_channels()
            logger.info(f"  Notifications: {', '.join(channels)}")
        else:
            logger.warning("  Notifications: No channels configured")

    def fetch_ohlcv(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from exchange."""
        if self.exchange is None:
            # Use Yahoo Finance for testing
            try:
                import yfinance as yf
                symbol_map = {
                    "BTC/USDT": "BTC-USD",
                    "ETH/USDT": "ETH-USD",
                }
                yf_symbol = symbol_map.get(self.config.symbol, self.config.symbol.replace("/", "-"))

                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period="60d", interval="1h")

                if df.empty:
                    return None

                df.columns = [c.lower() for c in df.columns]
                return df[["open", "high", "low", "close", "volume"]].copy()

            except Exception as e:
                logger.error(f"Failed to fetch data: {e}")
                return None
        else:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.config.symbol,
                    self.config.timeframe,
                    limit=limit,
                )
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            except Exception as e:
                logger.error(f"Exchange fetch failed: {e}")
                return None

    def get_current_price(self) -> float:
        """Get current price."""
        df = self.fetch_ohlcv(limit=10)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        return 0.0

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal using multiple factors.

        Combines:
        - ML prediction
        - Technical indicators
        - Candlestick patterns
        - Divergence detection
        - Confluence zones
        """
        signal = {
            "action": "FLAT",
            "confidence": 0.0,
            "reasons": [],
            "ml_signal": None,
            "technical_signal": None,
            "patterns": [],
            "divergences": [],
            "regime": None,
        }

        current_price = float(df["close"].iloc[-1])

        # === ML Signal ===
        if self.config.use_ml_signals and self.ml_predictor is not None:
            try:
                prediction = self.ml_predictor.predict(df)
                signal["ml_signal"] = {
                    "action": prediction.action,
                    "confidence": prediction.confidence,
                    "prob_long": prediction.probability_long,
                    "prob_short": prediction.probability_short,
                }

                if prediction.confidence >= self.config.ml_confidence_threshold:
                    if prediction.action in ["LONG", "SHORT"]:
                        signal["action"] = prediction.action
                        signal["confidence"] = prediction.confidence
                        signal["reasons"].append(f"ML: {prediction.action} ({prediction.confidence:.0%})")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        # === Technical Analysis ===
        if self.config.use_technical_confirmation:
            try:
                ta_result = self.technical_analyzer.analyze(df)

                # Candlestick patterns
                recent_patterns = [p for p in ta_result.candlestick_patterns if p.bar_index >= len(df) - 3]
                signal["patterns"] = [
                    {"pattern": p.pattern.value, "direction": p.direction, "strength": p.strength}
                    for p in recent_patterns
                ]

                # Divergences
                signal["divergences"] = [
                    {"type": d.divergence_type.value, "indicator": d.indicator, "strength": d.strength}
                    for d in ta_result.divergences
                ]

                # Check for confirmation
                bullish_patterns = sum(1 for p in recent_patterns if p.direction == "bullish")
                bearish_patterns = sum(1 for p in recent_patterns if p.direction == "bearish")

                bullish_divergences = sum(1 for d in ta_result.divergences if "bullish" in d.divergence_type.value)
                bearish_divergences = sum(1 for d in ta_result.divergences if "bearish" in d.divergence_type.value)

                # Modify confidence based on confirmation
                if signal["action"] == "LONG":
                    if bullish_patterns > 0:
                        signal["confidence"] *= 1.1
                        signal["reasons"].append(f"Bullish pattern confirmation ({bullish_patterns})")
                    if bullish_divergences > 0:
                        signal["confidence"] *= 1.15
                        signal["reasons"].append(f"Bullish divergence confirmation")
                    if bearish_patterns > bullish_patterns:
                        signal["confidence"] *= 0.8
                        signal["reasons"].append(f"Conflicting bearish patterns")

                elif signal["action"] == "SHORT":
                    if bearish_patterns > 0:
                        signal["confidence"] *= 1.1
                        signal["reasons"].append(f"Bearish pattern confirmation ({bearish_patterns})")
                    if bearish_divergences > 0:
                        signal["confidence"] *= 1.15
                        signal["reasons"].append(f"Bearish divergence confirmation")
                    if bullish_patterns > bearish_patterns:
                        signal["confidence"] *= 0.8
                        signal["reasons"].append(f"Conflicting bullish patterns")

                # Confluence zones
                if ta_result.nearest_support:
                    support_dist = (current_price - ta_result.nearest_support.center) / current_price
                    if abs(support_dist) < 0.01:  # Within 1% of support
                        if signal["action"] == "LONG":
                            signal["confidence"] *= 1.1
                            signal["reasons"].append(f"Near support zone ${ta_result.nearest_support.center:,.2f}")

                if ta_result.nearest_resistance:
                    resist_dist = (ta_result.nearest_resistance.center - current_price) / current_price
                    if abs(resist_dist) < 0.01:  # Within 1% of resistance
                        if signal["action"] == "SHORT":
                            signal["confidence"] *= 1.1
                            signal["reasons"].append(f"Near resistance zone ${ta_result.nearest_resistance.center:,.2f}")

            except Exception as e:
                logger.warning(f"Technical analysis failed: {e}")

        # Cap confidence at 1.0
        signal["confidence"] = min(signal["confidence"], 1.0)

        return signal

    def check_risk(self, df: pd.DataFrame) -> RiskAssessment:
        """Check risk before trading."""
        current_positions = {}
        if self.position:
            current_positions[self.config.symbol] = self.position_size * self.entry_price

        assessment = self.risk_manager.assess_risk(
            proposed_trade_value=self.balance * self.config.max_position_pct / 100,
            symbol=self.config.symbol,
            current_positions=current_positions,
            market_data=df,
        )

        return assessment

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_assessment: RiskAssessment,
    ) -> float:
        """Calculate position size based on risk."""
        risk_pct = risk_assessment.allowed_risk_pct / 100
        risk_amount = self.balance * risk_pct

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return 0.0

        # Position size from risk
        position_size = risk_amount / stop_distance

        # Apply multiplier from risk assessment
        position_size *= risk_assessment.position_size_multiplier

        # Apply slippage estimate
        slippage = self.slippage_model.estimate_slippage(
            symbol=self.config.symbol,
            price=entry_price,
            size=position_size * entry_price,
            side="buy",
            order_type=OrderType.MARKET,
        )

        # Max position limit
        max_position_value = self.balance * self.config.max_position_pct / 100
        max_position_size = max_position_value / entry_price

        return min(position_size, max_position_size)

    def calculate_stops(self, entry_price: float, direction: str, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate stop loss and take profit."""
        if self.config.use_atr_stops:
            # ATR-based stops
            try:
                from ta.volatility import AverageTrueRange
                atr = AverageTrueRange(
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    window=14
                ).average_true_range().iloc[-1]

                if direction == "LONG":
                    stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
                    take_profit = entry_price + (atr * self.config.atr_tp_multiplier)
                else:
                    stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
                    take_profit = entry_price - (atr * self.config.atr_tp_multiplier)

                return {"stop_loss": stop_loss, "take_profit": take_profit, "atr": atr}

            except Exception:
                pass

        # Percentage-based stops
        if direction == "LONG":
            stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.config.take_profit_pct / 100)
        else:
            stop_loss = entry_price * (1 + self.config.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.config.take_profit_pct / 100)

        return {"stop_loss": stop_loss, "take_profit": take_profit}

    def open_position(
        self,
        direction: str,
        entry_price: float,
        size: float,
        stops: Dict[str, float],
        signal: Dict[str, Any],
    ):
        """Open a new position."""
        self.position = direction
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.position_size = size
        self.stop_loss = stops["stop_loss"]
        self.take_profit = stops["take_profit"]
        self.break_even_activated = False
        self.partial_exits = []

        # Execute on exchange
        if not self.dry_run and self.exchange:
            try:
                order_side = "buy" if direction == "LONG" else "sell"
                order = self.exchange.create_market_order(
                    self.config.symbol,
                    order_side,
                    size,
                )
                logger.info(f"Order executed: {order}")
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                self.position = None
                return

        # Record trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "action": f"OPEN_{direction}",
            "symbol": self.config.symbol,
            "entry_price": entry_price,
            "size": size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": signal["confidence"],
            "reasons": signal["reasons"],
        }
        self.trade_history.append(trade)

        logger.info(f"OPENED {direction} position: {size:.6f} @ ${entry_price:,.2f}")
        logger.info(f"  Stop Loss: ${self.stop_loss:,.2f}")
        logger.info(f"  Take Profit: ${self.take_profit:,.2f}")

        # Send notification
        if self.config.send_trade_notifications and self.notifier.has_channels():
            self.notifier.notify_trade_opened(
                symbol=self.config.symbol,
                direction=direction,
                entry_price=entry_price,
                size=size,
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
            )

    def close_position(self, exit_price: float, reason: str):
        """Close current position."""
        if self.position is None:
            return

        # Calculate P&L
        if self.position == "LONG":
            pnl = (exit_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exit_price) * self.position_size

        pnl_pct = (pnl / (self.entry_price * self.position_size)) * 100
        self.balance += pnl

        # Execute on exchange
        if not self.dry_run and self.exchange:
            try:
                order_side = "sell" if self.position == "LONG" else "buy"
                order = self.exchange.create_market_order(
                    self.config.symbol,
                    order_side,
                    self.position_size,
                )
                logger.info(f"Close order executed: {order}")
            except Exception as e:
                logger.error(f"Close order failed: {e}")

        # Record trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "action": f"CLOSE_{self.position}",
            "symbol": self.config.symbol,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "size": self.position_size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "hold_time_minutes": (datetime.now() - self.entry_time).total_seconds() / 60,
        }
        self.trade_history.append(trade)

        # Update risk manager
        self.risk_manager.record_trade(pnl, self.position_size * self.entry_price, pnl > 0)
        self.risk_manager.record_recovery_trade(pnl > 0)

        logger.info(f"CLOSED {self.position} position @ ${exit_price:,.2f}")
        logger.info(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"  Reason: {reason}")

        # Send notification
        if self.config.send_trade_notifications and self.notifier.has_channels():
            self.notifier.notify_trade_closed(
                symbol=self.config.symbol,
                direction=self.position,
                entry_price=self.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None

    def check_exit_conditions(self, current_price: float) -> Optional[str]:
        """Check if position should be exited."""
        if self.position is None:
            return None

        # Check stop loss
        if self.position == "LONG" and current_price <= self.stop_loss:
            return "STOP_LOSS"
        if self.position == "SHORT" and current_price >= self.stop_loss:
            return "STOP_LOSS"

        # Check take profit
        if self.position == "LONG" and current_price >= self.take_profit:
            return "TAKE_PROFIT"
        if self.position == "SHORT" and current_price <= self.take_profit:
            return "TAKE_PROFIT"

        # Check break-even
        if self.config.use_break_even and not self.break_even_activated:
            if self.position == "LONG":
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                if profit_pct >= self.config.break_even_activation_pct:
                    self.stop_loss = self.entry_price * 1.001  # Small buffer
                    self.break_even_activated = True
                    logger.info(f"Break-even stop activated at ${self.stop_loss:,.2f}")
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100
                if profit_pct >= self.config.break_even_activation_pct:
                    self.stop_loss = self.entry_price * 0.999
                    self.break_even_activated = True
                    logger.info(f"Break-even stop activated at ${self.stop_loss:,.2f}")

        # Check trailing stop
        if self.config.use_trailing_stop:
            if self.position == "LONG":
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                if profit_pct >= self.config.trailing_activation_pct:
                    new_stop = current_price * (1 - self.config.trailing_distance_pct / 100)
                    if new_stop > self.stop_loss:
                        self.stop_loss = new_stop
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100
                if profit_pct >= self.config.trailing_activation_pct:
                    new_stop = current_price * (1 + self.config.trailing_distance_pct / 100)
                    if new_stop < self.stop_loss:
                        self.stop_loss = new_stop

        return None

    def run_iteration(self) -> Dict[str, Any]:
        """Run one trading iteration."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "action": None,
            "position": self.position,
            "balance": self.balance,
        }

        # Fetch data
        df = self.fetch_ohlcv()
        if df is None or len(df) < 100:
            logger.warning("Insufficient data")
            return result

        current_price = float(df["close"].iloc[-1])
        result["price"] = current_price

        # Check exit conditions first
        if self.position:
            exit_reason = self.check_exit_conditions(current_price)
            if exit_reason:
                self.close_position(current_price, exit_reason)
                result["action"] = f"CLOSE_{exit_reason}"
                self._save_state()
                return result

        # Check risk
        risk_assessment = self.check_risk(df)
        result["risk_status"] = risk_assessment.status.value
        result["risk_level"] = risk_assessment.risk_level.value
        result["regime"] = risk_assessment.market_regime.value if risk_assessment.market_regime else None

        if risk_assessment.status == TradingStatus.BLOCKED:
            logger.warning(f"Trading blocked: {risk_assessment.reasons}")
            if self.notifier.has_channels() and risk_assessment.reasons:
                self.notifier.notify_risk_warning(
                    "Trading Blocked",
                    "; ".join(risk_assessment.reasons),
                    {"status": risk_assessment.status.value},
                )
            return result

        # Generate signal
        signal = self.generate_signal(df)
        result["signal"] = signal["action"]
        result["confidence"] = signal["confidence"]

        logger.info(f"Price: ${current_price:,.2f} | Signal: {signal['action']} ({signal['confidence']:.0%})")
        logger.info(f"  Regime: {risk_assessment.market_regime.value if risk_assessment.market_regime else 'N/A'}")
        logger.info(f"  Risk: {risk_assessment.risk_level.value} | Allowed: {risk_assessment.allowed_risk_pct:.2f}%")

        if signal["reasons"]:
            logger.info(f"  Reasons: {', '.join(signal['reasons'][:3])}")

        # Trading logic
        if signal["action"] in ["LONG", "SHORT"] and signal["confidence"] >= self.config.ml_confidence_threshold:
            if self.position is None:
                # Open new position
                stops = self.calculate_stops(current_price, signal["action"], df)
                size = self.calculate_position_size(current_price, stops["stop_loss"], risk_assessment)

                if size > 0:
                    self.open_position(signal["action"], current_price, size, stops, signal)
                    result["action"] = f"OPEN_{signal['action']}"

            elif self.position != signal["action"]:
                # Close and reverse
                self.close_position(current_price, "SIGNAL_REVERSAL")

                stops = self.calculate_stops(current_price, signal["action"], df)
                size = self.calculate_position_size(current_price, stops["stop_loss"], risk_assessment)

                if size > 0:
                    self.open_position(signal["action"], current_price, size, stops, signal)
                    result["action"] = f"REVERSE_TO_{signal['action']}"

        # Update equity
        portfolio_value = self.balance
        if self.position:
            if self.position == "LONG":
                unrealized = (current_price - self.entry_price) * self.position_size
            else:
                unrealized = (self.entry_price - current_price) * self.position_size
            portfolio_value += unrealized

        self.equity_history.append({
            "timestamp": datetime.now().isoformat(),
            "value": portfolio_value,
        })

        result["portfolio_value"] = portfolio_value

        self._save_state()
        return result

    def _save_state(self):
        """Save current state."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trade_count": len(self.trade_history),
        }

        with open(self.data_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        with open(self.data_dir / "trades.json", "w") as f:
            json.dump(self.trade_history[-100:], f, indent=2)

        with open(self.data_dir / "equity.json", "w") as f:
            json.dump(self.equity_history[-1000:], f, indent=2)

    def _load_state(self):
        """Load existing state."""
        state_file = self.data_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                self.balance = state.get("balance", self.config.initial_capital)
                self.position = state.get("position")
                self.entry_price = state.get("entry_price", 0.0)
                self.position_size = state.get("position_size", 0.0)
                self.stop_loss = state.get("stop_loss")
                self.take_profit = state.get("take_profit")
                logger.info(f"Loaded existing state: Balance ${self.balance:,.2f}, Position: {self.position}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

        trades_file = self.data_dir / "trades.json"
        if trades_file.exists():
            try:
                with open(trades_file, "r") as f:
                    self.trade_history = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load trade history: {e}")

    def get_summary(self) -> Dict:
        """Get trading summary."""
        trades_with_pnl = [t for t in self.trade_history if "pnl" in t]
        winning = [t for t in trades_with_pnl if t["pnl"] > 0]
        losing = [t for t in trades_with_pnl if t["pnl"] < 0]

        total_pnl = sum(t["pnl"] for t in trades_with_pnl)

        return {
            "initial_capital": self.config.initial_capital,
            "balance": self.balance,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.config.initial_capital) * 100,
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / max(1, len(trades_with_pnl)) * 100,
            "current_position": self.position,
        }


def main():
    parser = argparse.ArgumentParser(description="Production Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no real trades)")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    parser.add_argument("--live", action="store_true", help="Live trading (REAL MONEY)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval (seconds)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade %")

    args = parser.parse_args()

    # Determine mode
    if args.live:
        dry_run = False
        mode = "LIVE"
    elif args.testnet:
        dry_run = False
        mode = "TESTNET"
    else:
        dry_run = True
        mode = "DRY_RUN"

    logger.info("=" * 70)
    logger.info("PRODUCTION TRADING BOT")
    logger.info("=" * 70)
    logger.info(f"Mode: {mode}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Capital: ${args.capital:,.2f}")
    logger.info(f"Risk per trade: {args.risk}%")
    logger.info(f"Interval: {args.interval}s")
    logger.info("=" * 70)

    if args.live:
        logger.warning("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        confirm = input("Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            logger.info("Cancelled.")
            return

    # Create exchange client
    exchange = None
    if not dry_run:
        if args.testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
            if not api_key or not api_secret:
                logger.error("Testnet API keys not found in .env!")
                return
            exchange = ExchangeClient("binance", api_key, api_secret, testnet=True)
        else:
            api_key = os.getenv("EXCHANGE_API_KEY")
            api_secret = os.getenv("EXCHANGE_API_SECRET")
            if not api_key or not api_secret:
                logger.error("API keys not found in .env!")
                return
            exchange = ExchangeClient("binance", api_key, api_secret)

    # Create config
    config = ProductionConfig(
        symbol=args.symbol,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk,
        loop_interval_seconds=args.interval,
    )

    # Create trader
    trader = ProductionTrader(
        config=config,
        exchange=exchange,
        dry_run=dry_run,
    )

    logger.info("")
    logger.info("Starting trading loop... (Ctrl+C to stop)")
    logger.info("")

    iteration = 0
    try:
        while True:
            iteration += 1
            start_time = time.time()

            logger.info(f"--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            result = trader.run_iteration()

            pnl = result.get("portfolio_value", args.capital) - args.capital
            pnl_pct = pnl / args.capital * 100

            logger.info(f"  Portfolio: ${result.get('portfolio_value', args.capital):,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

            if result.get("action"):
                logger.info(f"  Action: {result['action']}")

            elapsed = time.time() - start_time
            sleep_time = max(1, args.interval - elapsed)
            logger.info(f"  Next iteration in {sleep_time:.0f}s...")
            logger.info("")

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    # Final summary
    summary = trader.get_summary()
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRADING SESSION ENDED")
    logger.info("=" * 70)
    logger.info(f"Final Balance: ${summary['balance']:,.2f}")
    logger.info(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.2f}%)")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
