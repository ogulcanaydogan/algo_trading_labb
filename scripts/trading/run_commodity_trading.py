#!/usr/bin/env python3
"""
Commodity Trading Bot with Real Market Data + ML Framework.

Runs continuously, fetching real prices and making ML-informed trading decisions
for commodities: Gold, Silver, Crude Oil, and Natural Gas.
Uses the multi-asset engine with commodity-specific settings.
Sends Telegram alerts for buy/sell signals.

Based on run_live_paper_trading.py but configured for commodities.
Deep Learning predictions are disabled (no trained commodity models yet).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd

# Data directory for commodity trading
DATA_DIR = Path("data/commodity_trading")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "commodity_trading.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("commodity-trading")

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

# Import data store for persistent recording
try:
    from bot.data_store import (
        record_snapshot,
        record_signal,
    )
    HAS_DATA_STORE = True
    logger.info("Data Store initialized for persistent recording")
except ImportError:
    HAS_DATA_STORE = False

# System Logger for centralized event tracking
try:
    from bot.system_logger import (
        log_bot_start, log_bot_stop, heartbeat, EventType
    )
    HAS_SYSTEM_LOGGER = True
    logger.info("System Logger initialized for event tracking")
except ImportError:
    HAS_SYSTEM_LOGGER = False

# Intelligent Trading Brain for AI-driven decisions
try:
    from bot.intelligence import (
        IntelligentTradingBrain,
        BrainConfig,
        TradeOutcome,
        get_intelligent_brain,
    )
    HAS_INTELLIGENT_BRAIN = True
except ImportError as e:
    HAS_INTELLIGENT_BRAIN = False
    logging.debug(f"Intelligent Brain not available: {e}")


# Commodity Symbol Mapping
# Trading symbols -> Yahoo Finance symbols
SYMBOL_MAP = {
    "XAU/USD": "GC=F",    # Gold Futures
    "XAG/USD": "SI=F",    # Silver Futures
    "USOIL/USD": "CL=F",  # Crude Oil Futures
    "NATGAS/USD": "NG=F", # Natural Gas Futures
}

# Commodity display names
COMMODITY_NAMES = {
    "XAU/USD": "Gold",
    "XAG/USD": "Silver",
    "USOIL/USD": "Crude Oil",
    "NATGAS/USD": "Natural Gas",
}

# Load unified configuration
from bot.config import load_config
app_config = load_config()

# Configuration - prefer env vars, then unified config
LOOP_INTERVAL = int(os.getenv("COMMODITY_LOOP_INTERVAL", str(app_config.trading.loop_interval)))  # seconds
INITIAL_CAPITAL = float(os.getenv("COMMODITY_INITIAL_CAPITAL", str(app_config.trading.initial_capital)))
REBALANCE_THRESHOLD = float(os.getenv("COMMODITY_REBALANCE_THRESHOLD", str(app_config.trading.rebalance_threshold)))
SYMBOL_FETCH_DELAY = app_config.trading.symbol_fetch_delay  # Delay between fetching each symbol

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# ML Configuration (DL predictions disabled for commodities - no trained models yet)
USE_ML_SIGNALS = os.getenv("COMMODITY_USE_ML_SIGNALS", "true").lower() == "true"
USE_DEEP_LEARNING = False  # Disabled - no trained commodity models yet


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
):
    """Send a trade alert via Telegram."""
    if not manager.has_channels():
        return

    alert_type = AlertType.TRADE_OPENED if action == "BUY" else AlertType.TRADE_CLOSED
    commodity_name = COMMODITY_NAMES.get(symbol, symbol)

    alert = Alert(
        alert_type=alert_type,
        level=AlertLevel.ALERT,
        title=f"{action} Signal: {commodity_name} ({symbol})",
        message=f"Executed {action} order for {commodity_name}\n\n"
                f"Quantity: {quantity:.4f}\n"
                f"Price: ${price:,.2f}\n"
                f"Value: ${quantity * price:,.2f}\n"
                f"Regime: {regime}\n"
                f"Confidence: {confidence:.1%}",
        data={
            "symbol": symbol,
            "commodity": commodity_name,
            "action": action,
            "quantity": quantity,
            "price": price,
            "regime": regime,
            "confidence": confidence,
            "portfolio_value": portfolio_value,
        },
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

    commodity_name = COMMODITY_NAMES.get(symbol, symbol)

    alert = Alert(
        alert_type=AlertType.SIGNAL_GENERATED,
        level=AlertLevel.INFO,
        title=f"{action} Signal: {commodity_name}",
        message=f"ML model detected {action} opportunity\n\n"
                f"Commodity: {commodity_name} ({symbol})\n"
                f"Current Price: ${price:,.2f}\n"
                f"Market Regime: {regime}\n"
                f"Confidence: {confidence:.1%}",
        data={
            "symbol": symbol,
            "commodity": commodity_name,
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

    commodities_list = ", ".join(COMMODITY_NAMES.values())

    alert = Alert(
        alert_type=AlertType.SYSTEM,
        level=AlertLevel.INFO,
        title="Commodity Trading Bot Started",
        message=f"Commodity paper trading session started\n\n"
                f"Initial Capital: ${capital:,.2f}\n"
                f"Assets: {commodities_list}\n"
                f"Strategy: Risk Parity + ML Regime Detection\n"
                f"Rebalance Threshold: {REBALANCE_THRESHOLD:.0%}",
        data={"capital": capital, "loop_interval": LOOP_INTERVAL},
    )

    manager.send_alert(alert)


def send_shutdown_alert(manager: NotificationManager, portfolio_value: float, pnl: float):
    """Send shutdown notification."""
    if not manager.has_channels():
        return

    pnl_status = "Profit" if pnl >= 0 else "Loss"

    alert = Alert(
        alert_type=AlertType.SYSTEM,
        level=AlertLevel.WARNING,
        title="Commodity Trading Bot Stopped",
        message=f"Commodity trading session ended\n\n"
                f"Final Value: ${portfolio_value:,.2f}\n"
                f"P&L: {pnl_status} ${abs(pnl):,.2f} ({pnl/INITIAL_CAPITAL*100:+.2f}%)",
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
    cache_key = f"commodity:{yf_symbol}:{period}:{interval}"
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
            logger.warning(f"No data returned for {symbol} ({yf_symbol})")
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
        logger.warning(f"Failed to fetch {symbol} ({yf_symbol}): {e}")
        return None


def run_trading_loop():
    """Main trading loop for commodities."""
    logger.info("=" * 60)
    logger.info("STARTING COMMODITY TRADING BOT")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Loop Interval: {LOOP_INTERVAL}s")
    logger.info(f"Rebalance Threshold: {REBALANCE_THRESHOLD:.0%}")
    logger.info(f"Telegram Alerts: {'Enabled' if TELEGRAM_ENABLED else 'Disabled'}")
    logger.info(f"ML Signals: {'Enabled' if USE_ML_SIGNALS else 'Disabled'}")
    logger.info(f"Deep Learning: Disabled (no trained commodity models)")
    logger.info("-" * 60)
    logger.info("Commodities:")
    for trading_symbol, yf_symbol in SYMBOL_MAP.items():
        name = COMMODITY_NAMES.get(trading_symbol, trading_symbol)
        logger.info(f"  {name}: {trading_symbol} -> {yf_symbol}")
    logger.info("=" * 60)

    # Setup notifications
    notifier = setup_notifications()

    # Configure portfolio for commodities
    # Commodity-specific weight constraints based on volatility and liquidity
    config = MultiAssetConfig(
        assets=[
            # Gold - most stable, allow higher weight
            AssetConfig(
                symbol="XAU/USD",
                asset_type="commodity",
                max_weight=0.40,
                min_weight=0.15,
            ),
            # Silver - more volatile than gold
            AssetConfig(
                symbol="XAG/USD",
                asset_type="commodity",
                max_weight=0.30,
                min_weight=0.10,
            ),
            # Crude Oil - high volatility
            AssetConfig(
                symbol="USOIL/USD",
                asset_type="commodity",
                max_weight=0.30,
                min_weight=0.10,
            ),
            # Natural Gas - highest volatility, lower allocation
            AssetConfig(
                symbol="NATGAS/USD",
                asset_type="commodity",
                max_weight=0.25,
                min_weight=0.05,
            ),
        ],
        optimization_method=OptimizationMethod.RISK_PARITY,
        rebalance_threshold=REBALANCE_THRESHOLD,
        rebalance_frequency="daily",
        total_capital=INITIAL_CAPITAL,
        use_correlation_filter=True,  # Important for commodities (gold/silver correlation)
        max_correlation=0.90,  # Allow slightly higher correlation for precious metals
        risk_free_rate=0.04,  # Current risk-free rate
    )

    # Initialize engine
    engine = MultiAssetTradingEngine(
        config=config,
        data_dir=str(DATA_DIR),
    )

    # Initialize Intelligent Trading Brain
    intelligent_brain = None
    if HAS_INTELLIGENT_BRAIN:
        try:
            intelligent_brain = get_intelligent_brain()
            logger.info("Intelligent Trading Brain initialized - AI explanations enabled")
        except Exception as e:
            logger.warning(f"Could not initialize intelligent brain: {e}")

    # Send startup alert
    send_startup_alert(notifier, INITIAL_CAPITAL)

    # Register with system logger
    if HAS_SYSTEM_LOGGER:
        log_bot_start("commodity_trading", "commodity", {
            "initial_capital": INITIAL_CAPITAL,
            "loop_interval": LOOP_INTERVAL,
            "commodities": list(SYMBOL_MAP.keys()),
            "ml_signals": USE_ML_SIGNALS,
            "telegram_enabled": TELEGRAM_ENABLED
        })

    iteration = 0
    last_prices = {}
    last_signals = {}  # Track last signals to avoid duplicate alerts

    while True:
        iteration += 1
        loop_start = time.time()

        # Send heartbeat to system logger
        if HAS_SYSTEM_LOGGER:
            heartbeat("commodity_trading")

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

                arrow = "^" if change_pct > 0.1 else "v" if change_pct < -0.1 else "-"
                name = COMMODITY_NAMES.get(symbol, symbol)

                logger.info(f"  {name:12} ({symbol}): ${current_price:,.2f} {arrow} ({change_pct:+.2f}%)")
                last_prices[symbol] = current_price

            # Analyze portfolio
            decision = engine.analyze_portfolio(market_data)

            # Log ML decisions and send signal alerts for new signals
            for ad in decision.asset_decisions:
                signal = "LONG" if ad.action == "LONG" else "SHORT" if ad.action == "SHORT" else "FLAT"
                name = COMMODITY_NAMES.get(ad.symbol, ad.symbol)

                logger.info(f"  {name:12}: {signal:5} | {ad.regime:10} | conf: {ad.confidence:.0%}")

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

                    # Use Intelligent Brain to explain the signal
                    if intelligent_brain and HAS_INTELLIGENT_BRAIN:
                        try:
                            intelligent_brain.explain_entry(
                                symbol=ad.symbol,
                                action="BUY" if ad.action == "LONG" else "SELL",
                                price=last_prices.get(ad.symbol, 0),
                                confidence=ad.confidence,
                                regime=ad.regime,
                                signal_reason=f"ML signal with {ad.confidence:.0%} confidence",
                                position_size=decision.total_value * 0.2,  # Approximate
                                portfolio_value=decision.total_value,
                            )
                        except Exception as e:
                            logger.debug(f"Brain explain error: {e}")

                last_signals[ad.symbol] = ad.action

            # Log portfolio status
            logger.info(f"  Portfolio: ${decision.total_value:,.2f} | Cash: ${decision.cash_balance:,.2f}")

            # Execute rebalancing if needed
            if decision.rebalance_needed and decision.trades_to_execute:
                logger.info("  Rebalancing portfolio...")
                result = engine.execute_rebalance(decision)

                for trade in result.get("trades", []):
                    if trade.get("status") == "executed":
                        logger.info(f"    {trade['action']} {trade['quantity']:.4f} {trade['symbol']} @ ${trade['price']:,.2f}")

                        # Find regime and confidence for this trade
                        trade_ad = next(
                            (ad for ad in decision.asset_decisions if ad.symbol == trade["symbol"]),
                            None
                        )
                        regime = trade_ad.regime if trade_ad else "unknown"
                        confidence = trade_ad.confidence if trade_ad else 0

                        # Send trade alert via Telegram
                        send_trade_alert(
                            notifier,
                            action=trade["action"],
                            symbol=trade["symbol"],
                            quantity=trade["quantity"],
                            price=trade["price"],
                            regime=regime,
                            confidence=confidence,
                            portfolio_value=result.get("portfolio_value", 0),
                        )

                logger.info(f"  New portfolio value: ${result.get('portfolio_value', 0):,.2f}")
            else:
                logger.info("  No rebalancing needed")

            # Save state every iteration for dashboard
            status = engine.get_portfolio_status()
            state_file = DATA_DIR / "state.json"
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "market_type": "commodity",
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
                    }
                    for ad in decision.asset_decisions
                },
                "commodities": {
                    symbol: {
                        "name": COMMODITY_NAMES.get(symbol, symbol),
                        "yf_symbol": SYMBOL_MAP.get(symbol),
                        "price": last_prices.get(symbol, 0),
                    }
                    for symbol in SYMBOL_MAP.keys()
                },
                "deep_learning_enabled": False,
                "ml_signals_enabled": USE_ML_SIGNALS,
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            # Log checkpoint every 10 iterations
            if iteration % 10 == 0:
                logger.info(f"  [Checkpoint] Total: ${status['total_value']:,.2f}, Positions: {len(status['positions'])}")

            # Record snapshot and signals to data store
            if HAS_DATA_STORE and iteration % 10 == 0:
                try:
                    record_snapshot(
                        total_value=status["total_value"],
                        cash_balance=status["cash_balance"],
                        positions=status["positions"],
                        pnl=status["total_value"] - INITIAL_CAPITAL,
                        pnl_pct=(status["total_value"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
                        market_values={"commodity": status["total_value"]},
                    )
                    # Record signals
                    for ad in decision.asset_decisions:
                        record_signal(
                            symbol=ad.symbol,
                            market="commodity",
                            signal=ad.action,
                            regime=ad.regime,
                            confidence=ad.confidence,
                            price=last_prices.get(ad.symbol, 0),
                        )
                except Exception as e:
                    logger.warning(f"Failed to record to data store: {e}")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            break
        except Exception as e:
            logger.exception(f"Error in trading loop: {e}")

        # Wait for next iteration
        elapsed = time.time() - loop_start
        sleep_time = max(1, LOOP_INTERVAL - elapsed)
        time.sleep(sleep_time)

    # Unregister from system logger
    if HAS_SYSTEM_LOGGER:
        log_bot_stop("commodity_trading", "normal_shutdown")

    # Final summary
    logger.info("=" * 60)
    logger.info("COMMODITY TRADING SESSION ENDED")
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
            name = COMMODITY_NAMES.get(symbol, symbol)
            logger.info(f"  {name} ({symbol}): {pos['quantity']:.4f} @ ${pos['price']:,.2f} = ${pos['value']:,.2f}")

        # Send shutdown alert
        send_shutdown_alert(notifier, portfolio_value, pnl)

    except Exception as e:
        logger.error(f"Error getting final status: {e}")


if __name__ == "__main__":
    run_trading_loop()
