#!/usr/bin/env python3
"""
Live Paper Trading with Real Market Data + Deep Learning Models.

Runs continuously, fetching real prices and making ML-informed trading decisions.
Uses LSTM/Transformer models with regime-based selection.
Sends Telegram alerts for buy/sell signals.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd

# Load unified configuration
from bot.config import load_config, get_data_dir

# Load configuration (env vars override config.yaml)
app_config = load_config()

# Setup logging using config
LOG_DIR = get_data_dir() / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_level = getattr(logging, app_config.general.log_level, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "paper_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("paper-trading")

# Suppress noisy loggers
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

try:
    import yfinance as yf
except ImportError:
    logger.error("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

from bot.multi_asset_engine import (
    MultiAssetTradingEngine,
    MultiAssetConfig,
    AssetConfig,
)
from bot.portfolio_optimizer import OptimizationMethod
from bot.notifications import (
    NotificationManager,
    TelegramChannel,
    Alert,
    AlertType,
    AlertLevel,
)

# Data Store for persistent trade/snapshot recording
try:
    from bot.data_store import get_data_store, record_trade, record_snapshot, record_signal
    HAS_DATA_STORE = True
    logger.info("Data Store initialized for persistent recording")
except ImportError:
    HAS_DATA_STORE = False
    logger.warning("Data Store not available - trades won't be persisted")

# System Logger for centralized event tracking
try:
    from bot.system_logger import (
        get_system_logger, log_bot_start, log_bot_stop, log_trade,
        log_event, log_error, heartbeat, EventType, Severity
    )
    HAS_SYSTEM_LOGGER = True
except ImportError:
    HAS_SYSTEM_LOGGER = False
    logger.warning("System Logger not available")

# Intelligent Trading Brain for AI-driven decisions
try:
    from bot.intelligence import (
        IntelligentTradingBrain,
        BrainConfig,
        TradeOutcome,
    )
    HAS_INTELLIGENT_BRAIN = True
    logger.info("Intelligent Trading Brain available")
except ImportError as e:
    HAS_INTELLIGENT_BRAIN = False
    logger.warning(f"Intelligent Trading Brain not available: {e}")

# Deep Learning models
try:
    from bot.ml.feature_engineer import FeatureEngineer
    from bot.ml.registry import ModelRegistry, ModelSelector, ModelSelectionStrategy
    from sklearn.preprocessing import StandardScaler
    HAS_DL_MODELS = True
    logger.info("Deep Learning models available")
except ImportError as e:
    HAS_DL_MODELS = False
    logger.warning(f"Deep Learning models not available: {e}")


# Configuration from unified config system
# Build SYMBOL_MAP from config
SYMBOL_MAP = {symbol: symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
              for symbol in app_config.crypto.symbols}

LOOP_INTERVAL = app_config.trading.loop_interval
INITIAL_CAPITAL = app_config.trading.initial_capital
REBALANCE_THRESHOLD = app_config.trading.rebalance_threshold
SYMBOL_FETCH_DELAY = app_config.trading.symbol_fetch_delay

# Telegram Configuration (still needs env vars for secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = app_config.notifications.telegram_enabled and bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Deep Learning Configuration
USE_DEEP_LEARNING = app_config.deep_learning.enabled
DL_MODEL_SELECTION = app_config.deep_learning.model_selection

# AI Explanation Configuration (disabled by default to save API costs)
ENABLE_AI_EXPLANATIONS = app_config.notifications.ai_explanations

# AI Market Analyst for trade explanations
try:
    from bot.llm.claude import MarketAnalyst
    HAS_AI_ANALYST = True
except ImportError as e:
    HAS_AI_ANALYST = False
    logger.warning(f"AI Market Analyst not available: {e}")


class DeepLearningPredictor:
    """Wrapper for deep learning model predictions."""

    def __init__(self):
        self.registry = ModelRegistry(registry_dir="data/model_registry")
        strategy_map = {
            "regime_based": ModelSelectionStrategy.REGIME_BASED,
            "ensemble": ModelSelectionStrategy.ENSEMBLE,
            "single_best": ModelSelectionStrategy.SINGLE_BEST,
        }
        self.selector = ModelSelector(
            registry=self.registry,
            strategy=strategy_map.get(DL_MODEL_SELECTION, ModelSelectionStrategy.REGIME_BASED),
        )
        self.feature_engineer = FeatureEngineer()
        self.scalers = {}  # Per-symbol scalers

    def predict(self, symbol: str, ohlcv: pd.DataFrame, regime: str = "sideways"):
        """
        Make prediction using deep learning model.

        Returns:
            dict with action, confidence, model_type, probabilities
        """
        try:
            # Extract features
            df_features = self.feature_engineer.extract_features(ohlcv)
            if len(df_features) < 120:
                return None

            exclude_cols = ["target_return", "target_direction", "target_class",
                          "open", "high", "low", "close", "volume"]
            feature_cols = [c for c in df_features.columns if c not in exclude_cols]

            X = df_features[feature_cols].values

            # Handle inf/nan
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e6, 1e6)

            # Scale (use cached scaler or create new)
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()
                X_scaled = self.scalers[symbol].fit_transform(X)
            else:
                X_scaled = self.scalers[symbol].transform(X)

            # Get prediction from model selector
            prediction = self.selector.get_prediction(
                symbol=symbol,
                market_type="crypto",
                X=X_scaled,
                regime=regime,
            )

            return {
                "action": prediction.action,
                "confidence": prediction.confidence,
                "model_type": prediction.model_type,
                "prob_long": prediction.probability_long,
                "prob_short": prediction.probability_short,
                "prob_flat": prediction.probability_flat,
            }

        except Exception as e:
            logger.warning(f"DL prediction failed for {symbol}: {e}")
            return None

    def get_available_models(self, symbol: str):
        """Get list of available models for a symbol."""
        models = self.registry.list_models(symbol=symbol, market_type="crypto")
        return [{"type": m.model_type, "accuracy": m.val_accuracy} for m in models]


def generate_ai_enhanced_alert_message(
    action: str,
    symbol: str,
    quantity: float,
    price: float,
    regime: str,
    confidence: float,
    portfolio_value: float,
    ai_explanation: Optional[str] = None,
    dl_prediction: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate an AI-enhanced alert message for trade notifications.

    Args:
        action: Trade action (BUY/SELL)
        symbol: Trading symbol
        quantity: Trade quantity
        price: Trade price
        regime: Market regime
        confidence: Signal confidence
        portfolio_value: Current portfolio value
        ai_explanation: Optional AI-generated explanation of the trade
        dl_prediction: Optional deep learning prediction details

    Returns:
        Formatted alert message string with AI enhancements if available
    """
    # Base trade details
    message_parts = [
        f"Executed {action} order for {symbol}",
        "",
        f"Quantity: {quantity:.6f}",
        f"Price: ${price:,.2f}",
        f"Value: ${quantity * price:,.2f}",
        f"Regime: {regime}",
        f"Confidence: {confidence:.1%}",
    ]

    # Add model type if DL prediction is available
    if dl_prediction:
        model_type = dl_prediction.get("model_type", "unknown")
        dl_confidence = dl_prediction.get("confidence", 0)
        dl_action = dl_prediction.get("action", "N/A")
        message_parts.append("")
        message_parts.append(f"DL Model: {model_type}")
        message_parts.append(f"DL Signal: {dl_action} ({dl_confidence:.0%})")

    # Add AI explanation if available
    if ai_explanation:
        message_parts.append("")
        message_parts.append("AI Analysis:")
        message_parts.append(ai_explanation)

    return "\n".join(message_parts)


def get_ai_trade_explanation(
    analyst: "MarketAnalyst",
    symbol: str,
    action: str,
    price: float,
    regime: str,
    confidence: float,
    indicators: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Get an AI-generated explanation for a trade decision.

    Wraps the MarketAnalyst.explain_trade_briefly() call with error handling
    to ensure AI failures don't break the trading loop.

    Args:
        analyst: MarketAnalyst instance
        symbol: Trading symbol
        action: Trade action (BUY/SELL)
        price: Trade price
        regime: Market regime
        confidence: Signal confidence
        indicators: Optional technical indicators

    Returns:
        Brief AI explanation string or None if unavailable/error
    """
    try:
        explanation = analyst.explain_trade_briefly(
            symbol=symbol,
            action=action,
            price=price,
            regime=regime,
            confidence=confidence,
            indicators=indicators,
        )
        return explanation
    except Exception as e:
        logger.warning(f"AI explanation failed for {symbol} {action}: {e}")
        return None


def setup_notifications() -> NotificationManager:
    """Setup notification manager with Telegram."""
    manager = NotificationManager()

    if TELEGRAM_ENABLED:
        telegram = TelegramChannel(
            bot_token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
        )
        manager.add_channel("telegram", telegram)
        logger.info("Telegram notifications enabled")
    else:
        logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

    return manager


def send_trade_alert(
    manager: NotificationManager,
    action: str,
    symbol: str,
    quantity: float,
    price: float,
    regime: str,
    confidence: float,
    portfolio_value: float,
    ai_explanation: Optional[str] = None,
    dl_prediction: Optional[Dict[str, Any]] = None,
):
    """
    Send a trade alert via Telegram with optional AI enhancements.

    Args:
        manager: NotificationManager instance
        action: Trade action (BUY/SELL)
        symbol: Trading symbol
        quantity: Trade quantity
        price: Trade price
        regime: Market regime
        confidence: Signal confidence
        portfolio_value: Current portfolio value
        ai_explanation: Optional AI-generated trade explanation
        dl_prediction: Optional deep learning prediction details
    """
    if not manager.has_channels():
        return

    emoji = "🟢" if action == "BUY" else "🔴"
    alert_type = AlertType.TRADE_OPENED if action == "BUY" else AlertType.TRADE_CLOSED

    # Generate enhanced message with AI explanation if available
    message = generate_ai_enhanced_alert_message(
        action=action,
        symbol=symbol,
        quantity=quantity,
        price=price,
        regime=regime,
        confidence=confidence,
        portfolio_value=portfolio_value,
        ai_explanation=ai_explanation,
        dl_prediction=dl_prediction,
    )

    # Build data dict with optional DL info
    data = {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "regime": regime,
        "confidence": confidence,
        "portfolio_value": portfolio_value,
    }

    if dl_prediction:
        data["dl_model_type"] = dl_prediction.get("model_type")
        data["dl_confidence"] = dl_prediction.get("confidence")

    alert = Alert(
        alert_type=alert_type,
        level=AlertLevel.ALERT,
        title=f"{emoji} {action} Signal: {symbol}",
        message=message,
        data=data,
    )

    manager.send_alert(alert)


def send_signal_alert(
    manager: NotificationManager,
    symbol: str,
    action: str,
    regime: str,
    confidence: float,
    price: float,
):
    """Send a signal alert (before trade execution)."""
    if not manager.has_channels():
        return

    if action == "FLAT":
        return  # Don't alert on FLAT signals

    emoji = "📈" if action == "LONG" else "📉"

    alert = Alert(
        alert_type=AlertType.SIGNAL_GENERATED,
        level=AlertLevel.INFO,
        title=f"{emoji} {action} Signal: {symbol}",
        message=f"ML model detected {action} opportunity\n\n"
                f"Symbol: {symbol}\n"
                f"Current Price: ${price:,.2f}\n"
                f"Market Regime: {regime}\n"
                f"Confidence: {confidence:.1%}",
        data={
            "symbol": symbol,
            "action": action,
            "regime": regime,
            "confidence": confidence,
            "price": price,
        },
    )

    manager.send_alert(alert)


def send_startup_alert(manager: NotificationManager, capital: float):
    """Send startup notification."""
    if not manager.has_channels():
        return

    alert = Alert(
        alert_type=AlertType.SYSTEM,
        level=AlertLevel.INFO,
        title="🚀 Paper Trading Bot Started",
        message=f"Live paper trading session started\n\n"
                f"Initial Capital: ${capital:,.2f}\n"
                f"Assets: BTC, ETH, SOL, AVAX\n"
                f"Strategy: Risk Parity + ML Regime Detection\n"
                f"Rebalance Threshold: {REBALANCE_THRESHOLD:.0%}",
        data={"capital": capital, "loop_interval": LOOP_INTERVAL},
    )

    manager.send_alert(alert)


def send_shutdown_alert(manager: NotificationManager, portfolio_value: float, pnl: float):
    """Send shutdown notification."""
    if not manager.has_channels():
        return

    pnl_emoji = "📈" if pnl >= 0 else "📉"

    alert = Alert(
        alert_type=AlertType.SYSTEM,
        level=AlertLevel.WARNING,
        title="🛑 Paper Trading Bot Stopped",
        message=f"Paper trading session ended\n\n"
                f"Final Value: ${portfolio_value:,.2f}\n"
                f"P&L: {pnl_emoji} ${pnl:,.2f} ({pnl/INITIAL_CAPITAL*100:+.2f}%)",
        data={"portfolio_value": portfolio_value, "pnl": pnl},
    )

    manager.send_alert(alert)


# Initialize rate limiter for Yahoo Finance
try:
    from bot.rate_limiter import get_yahoo_rate_limiter, MultiLevelCache
    _rate_limiter = get_yahoo_rate_limiter()
    _cache = MultiLevelCache(
        cache_dir=Path("./data/cache"),
        memory_max_size=100,
        memory_ttl=60.0,
        disk_ttl=300.0,
    )
    logger.info("Rate limiter and cache initialized")
except ImportError:
    _rate_limiter = None
    _cache = None
    logger.warning("Rate limiter not available, using simple delays")


def fetch_market_data(symbol: str, period: str = "60d", interval: str = "1h"):
    """Fetch real market data from Yahoo Finance with rate limiting and caching."""
    yf_symbol = SYMBOL_MAP.get(symbol)
    if not yf_symbol:
        return None

    # Check cache first
    cache_key = f"crypto:{yf_symbol}:{period}:{interval}"
    if _cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {symbol}")
            import pandas as pd
            return pd.DataFrame(cached)

    # Acquire rate limit
    if _rate_limiter:
        if not _rate_limiter.acquire(block=True, timeout=120.0):
            logger.warning(f"Rate limit blocked for {symbol}")
            return None
    else:
        time.sleep(5)

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)

        if _rate_limiter:
            _rate_limiter.report_success()

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        result = df[["open", "high", "low", "close", "volume"]]

        # Cache the result
        if _cache and not result.empty:
            cache_data = result.reset_index().to_dict(orient='records')
            _cache.set(cache_key, cache_data)

        return result
    except Exception as e:
        if _rate_limiter:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "too many" in error_str
            _rate_limiter.report_error(is_rate_limit=is_rate_limit)
        logger.warning(f"Failed to fetch {symbol}: {e}")
        return None


def run_trading_loop():
    """Main trading loop."""
    logger.info("=" * 60)
    logger.info("STARTING LIVE PAPER TRADING + DEEP LEARNING")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Loop Interval: {LOOP_INTERVAL}s")
    logger.info(f"Rebalance Threshold: {REBALANCE_THRESHOLD:.0%}")
    logger.info(f"Telegram Alerts: {'Enabled' if TELEGRAM_ENABLED else 'Disabled'}")
    logger.info(f"Deep Learning: {'Enabled' if HAS_DL_MODELS and USE_DEEP_LEARNING else 'Disabled'}")
    if HAS_DL_MODELS and USE_DEEP_LEARNING:
        logger.info(f"Model Selection: {DL_MODEL_SELECTION}")
    logger.info(f"AI Explanations: {'Enabled' if ENABLE_AI_EXPLANATIONS and HAS_AI_ANALYST else 'Disabled'}")
    logger.info("=" * 60)

    # Log bot start to system logger
    if HAS_SYSTEM_LOGGER:
        log_bot_start("crypto_paper_trading", "crypto", {
            "initial_capital": INITIAL_CAPITAL,
            "loop_interval": LOOP_INTERVAL,
            "deep_learning": HAS_DL_MODELS and USE_DEEP_LEARNING,
            "telegram_enabled": TELEGRAM_ENABLED
        })

    # Setup notifications
    notifier = setup_notifications()

    # Configure portfolio
    config = MultiAssetConfig(
        assets=[
            AssetConfig(symbol="BTC/USDT", max_weight=0.35, min_weight=0.10),
            AssetConfig(symbol="ETH/USDT", max_weight=0.30, min_weight=0.10),
            AssetConfig(symbol="SOL/USDT", max_weight=0.25, min_weight=0.05),
            AssetConfig(symbol="AVAX/USDT", max_weight=0.20, min_weight=0.05),
        ],
        optimization_method=OptimizationMethod.RISK_PARITY,
        rebalance_threshold=REBALANCE_THRESHOLD,
        rebalance_frequency="daily",
        total_capital=INITIAL_CAPITAL,
        use_correlation_filter=False,
    )

    # Initialize engine
    engine = MultiAssetTradingEngine(
        config=config,
        data_dir="data/live_paper_trading",
    )

    # Initialize Deep Learning predictor
    dl_predictor = None
    if HAS_DL_MODELS and USE_DEEP_LEARNING:
        try:
            dl_predictor = DeepLearningPredictor()
            # Log available models
            for symbol in SYMBOL_MAP.keys():
                models = dl_predictor.get_available_models(symbol)
                if models:
                    model_info = ", ".join([f"{m['type']}({m['accuracy']:.0%})" for m in models])
                    logger.info(f"  {symbol} models: {model_info}")
                else:
                    logger.warning(f"  {symbol}: No trained models found")
        except Exception as e:
            logger.error(f"Failed to initialize DL predictor: {e}")
            dl_predictor = None

    # Initialize AI Market Analyst for trade explanations
    ai_analyst = None
    if ENABLE_AI_EXPLANATIONS and HAS_AI_ANALYST:
        try:
            ai_analyst = MarketAnalyst()
            logger.info("AI Market Analyst initialized for trade explanations")
        except Exception as e:
            logger.error(f"Failed to initialize AI analyst: {e}")
            ai_analyst = None

    # Initialize Intelligent Trading Brain
    intelligent_brain = None
    if HAS_INTELLIGENT_BRAIN:
        try:
            brain_config = BrainConfig(
                daily_budget=5.0,
                enable_claude=True,
                enable_ollama=True,
                telegram_enabled=TELEGRAM_ENABLED,
                learning_rate=0.1,
            )
            intelligent_brain = IntelligentTradingBrain(config=brain_config)
            logger.info("Intelligent Trading Brain initialized")
            intelligent_brain.log_status()
        except Exception as e:
            logger.error(f"Failed to initialize Intelligent Brain: {e}")
            intelligent_brain = None

    # Send startup alert
    send_startup_alert(notifier, INITIAL_CAPITAL)

    iteration = 0
    last_prices = {}
    last_signals = {}  # Track last signals to avoid duplicate alerts
    dl_predictions = {}  # Store DL predictions per symbol

    while True:
        iteration += 1
        loop_start = time.time()

        # Send heartbeat to system logger
        if HAS_SYSTEM_LOGGER:
            heartbeat("crypto_paper_trading")

        try:
            logger.info(f"--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # Fetch market data with delay between symbols to avoid rate limiting
            market_data = {}
            for i, symbol in enumerate(SYMBOL_MAP.keys()):
                if i > 0:
                    time.sleep(SYMBOL_FETCH_DELAY)  # Delay between symbol fetches
                df = fetch_market_data(symbol)
                if df is not None and len(df) > 50:
                    market_data[symbol] = df

            if not market_data:
                logger.warning("No market data available, skipping iteration")
                time.sleep(LOOP_INTERVAL)
                continue

            # Log price changes
            for symbol, df in market_data.items():
                current_price = df["close"].iloc[-1]
                prev_price = last_prices.get(symbol, current_price)
                change_pct = (current_price - prev_price) / prev_price * 100 if prev_price else 0

                arrow = "↑" if change_pct > 0.1 else "↓" if change_pct < -0.1 else "→"
                logger.info(f"  {symbol}: ${current_price:,.2f} {arrow} ({change_pct:+.2f}%)")
                last_prices[symbol] = current_price

            # Analyze portfolio
            decision = engine.analyze_portfolio(market_data)

            # Get Deep Learning predictions for each symbol
            if dl_predictor:
                for symbol, df in market_data.items():
                    # Get regime from engine decision
                    ad = next((a for a in decision.asset_decisions if a.symbol == symbol), None)
                    regime = ad.regime if ad else "sideways"

                    dl_pred = dl_predictor.predict(symbol, df, regime=regime)
                    if dl_pred:
                        dl_predictions[symbol] = dl_pred

            # Log ML decisions and send signal alerts for new signals
            for ad in decision.asset_decisions:
                signal = "LONG 🟢" if ad.action == "LONG" else "SHORT 🔴" if ad.action == "SHORT" else "FLAT ⚪"

                # Check if we have DL prediction for this symbol
                dl_pred = dl_predictions.get(ad.symbol)
                if dl_pred:
                    dl_signal = "🟢" if dl_pred["action"] == "LONG" else "🔴" if dl_pred["action"] == "SHORT" else "⚪"
                    logger.info(
                        f"  {ad.symbol}: {signal} | {ad.regime} | "
                        f"DL:{dl_signal}{dl_pred['action']}({dl_pred['model_type']}) {dl_pred['confidence']:.0%}"
                    )
                else:
                    logger.info(f"  {ad.symbol}: {signal} | {ad.regime} | conf: {ad.confidence:.0%}")

                # Send alert if signal changed from FLAT to LONG/SHORT
                prev_signal = last_signals.get(ad.symbol, "FLAT")
                if ad.action != "FLAT" and prev_signal == "FLAT":
                    send_signal_alert(
                        notifier,
                        symbol=ad.symbol,
                        action=ad.action,
                        regime=ad.regime,
                        confidence=ad.confidence,
                        price=last_prices.get(ad.symbol, 0),
                    )
                last_signals[ad.symbol] = ad.action

            # Log portfolio status
            logger.info(f"  Portfolio: ${decision.total_value:,.2f} | Cash: ${decision.cash_balance:,.2f}")

            # Execute rebalancing if needed
            if decision.rebalance_needed and decision.trades_to_execute:
                logger.info("  Rebalancing portfolio...")
                result = engine.execute_rebalance(decision)

                for trade in result.get("trades", []):
                    if trade.get("status") == "executed":
                        logger.info(f"    {trade['action']} {trade['quantity']:.6f} {trade['symbol']} @ ${trade['price']:,.2f}")

                        # Find regime and confidence for this trade
                        trade_ad = next(
                            (ad for ad in decision.asset_decisions if ad.symbol == trade["symbol"]),
                            None
                        )
                        regime = trade_ad.regime if trade_ad else "unknown"
                        confidence = trade_ad.confidence if trade_ad else 0

                        # Get DL prediction for this symbol (if available)
                        trade_dl_prediction = dl_predictions.get(trade["symbol"])

                        # Get AI explanation if enabled (with try/except to prevent breaking loop)
                        ai_explanation = None
                        if ai_analyst:
                            try:
                                ai_explanation = get_ai_trade_explanation(
                                    analyst=ai_analyst,
                                    symbol=trade["symbol"],
                                    action=trade["action"],
                                    price=trade["price"],
                                    regime=regime,
                                    confidence=confidence,
                                )
                                if ai_explanation:
                                    logger.info(f"    AI: {ai_explanation[:80]}...")
                            except Exception as e:
                                logger.warning(f"    AI explanation error (non-fatal): {e}")

                        # Intelligent Brain explanation (enhanced with learning)
                        if intelligent_brain:
                            try:
                                brain_explanation = intelligent_brain.explain_entry(
                                    symbol=trade["symbol"],
                                    action=trade["action"],
                                    price=trade["price"],
                                    quantity=trade["quantity"],
                                    signal={"confidence": confidence, "reason": "ML signal"},
                                    portfolio_context={
                                        "total_value": result.get("portfolio_value", INITIAL_CAPITAL),
                                        "daily_pnl_pct": (result.get("portfolio_value", INITIAL_CAPITAL) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
                                    },
                                )
                                logger.info(f"    Brain: {brain_explanation.short_explanation[:60]}...")
                            except Exception as e:
                                logger.debug(f"    Brain explanation error: {e}")

                        # Send trade alert via Telegram with AI enhancements
                        send_trade_alert(
                            notifier,
                            action=trade["action"],
                            symbol=trade["symbol"],
                            quantity=trade["quantity"],
                            price=trade["price"],
                            regime=regime,
                            confidence=confidence,
                            portfolio_value=result.get("portfolio_value", 0),
                            ai_explanation=ai_explanation,
                            dl_prediction=trade_dl_prediction,
                        )

                logger.info(f"  New portfolio value: ${result.get('portfolio_value', 0):,.2f}")
            else:
                logger.info("  No rebalancing needed")

            # Save state every iteration for dashboard
            status = engine.get_portfolio_status()
            state_file = Path("data/live_paper_trading/state.json")
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "market_type": "crypto",
                "total_value": status["total_value"],
                "cash_balance": status["cash_balance"],
                "initial_capital": INITIAL_CAPITAL,
                "pnl": status["total_value"] - INITIAL_CAPITAL,
                "pnl_pct": (status["total_value"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
                "positions": status["positions"],
                "positions_count": len(status["positions"]),
                "current_weights": status["current_weights"],
                "optimization_method": status["optimization_method"],
                "signals": {
                    ad.symbol: {
                        "signal": ad.action,
                        "regime": ad.regime,
                        "confidence": ad.confidence,
                        "price": last_prices.get(ad.symbol, 0),
                        # Deep Learning prediction
                        "dl_prediction": dl_predictions.get(ad.symbol),
                    }
                    for ad in decision.asset_decisions
                },
                "deep_learning_enabled": dl_predictor is not None,
                "dl_model_selection": DL_MODEL_SELECTION if dl_predictor else None,
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            # Record portfolio snapshot to persistent data store (every 10 iterations ~ every 30 min)
            if HAS_DATA_STORE and iteration % 10 == 0:
                try:
                    record_snapshot(
                        total_value=status["total_value"],
                        cash_balance=status["cash_balance"],
                        positions=status["positions"],
                        pnl=status["total_value"] - INITIAL_CAPITAL,
                        pnl_pct=(status["total_value"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
                        market_values={"crypto": status["total_value"]},
                    )
                except Exception as e:
                    logger.warning(f"Failed to record snapshot: {e}")

            # Record signals to persistent store (every iteration)
            if HAS_DATA_STORE:
                try:
                    for ad in decision.asset_decisions:
                        record_signal(
                            symbol=ad.symbol,
                            market="crypto",
                            signal=ad.action,
                            regime=ad.regime,
                            confidence=ad.confidence,
                            price=last_prices.get(ad.symbol, 0),
                            dl_prediction=dl_predictions.get(ad.symbol),
                        )
                except Exception as e:
                    logger.debug(f"Failed to record signals: {e}")

            # Log checkpoint every 10 iterations
            if iteration % 10 == 0:
                logger.info(f"  [Checkpoint] Total: ${status['total_value']:,.2f}, Positions: {len(status['positions'])}")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            break
        except Exception as e:
            logger.exception(f"Error in trading loop: {e}")

        # Wait for next iteration
        elapsed = time.time() - loop_start
        sleep_time = max(1, LOOP_INTERVAL - elapsed)
        time.sleep(sleep_time)

    # Final summary
    logger.info("=" * 60)
    logger.info("PAPER TRADING SESSION ENDED")
    logger.info("=" * 60)

    try:
        status = engine.get_portfolio_status()
        portfolio_value = status["total_value"]
        pnl = portfolio_value - INITIAL_CAPITAL

        logger.info(f"Final Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Cash Balance: ${status['cash_balance']:,.2f}")
        logger.info(f"Total P&L: ${pnl:,.2f} ({pnl/INITIAL_CAPITAL*100:+.2f}%)")
        logger.info("Positions:")
        for symbol, pos in status["positions"].items():
            logger.info(f"  {symbol}: {pos['quantity']:.6f} @ ${pos['price']:,.2f} = ${pos['value']:,.2f}")

        # Send shutdown alert
        send_shutdown_alert(notifier, portfolio_value, pnl)

    except Exception as e:
        logger.error(f"Error getting final status: {e}")


if __name__ == "__main__":
    run_trading_loop()
