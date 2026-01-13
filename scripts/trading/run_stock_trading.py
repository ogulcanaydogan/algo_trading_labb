#!/usr/bin/env python3
"""
Stock Trading Bot with Real Market Data + ML Framework.

Runs continuously, fetching real stock prices and making ML-informed trading decisions.
Uses technical analysis and ML signals for regime detection.
Sends Telegram alerts for buy/sell signals.

Configured for US stocks: AAPL, MSFT, GOOGL, NVDA, AMZN
Initial Capital: $10,000
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

# Configuration
DATA_DIR = Path("data/stock_trading")
LOG_DIR = Path("data/logs")
STATE_FILE = DATA_DIR / "state.json"
LOG_FILE = LOG_DIR / "stock_trading.log"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("stock-trading")

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


# Stock Symbol Configuration
# Maps internal symbol names to Yahoo Finance tickers
SYMBOL_MAP = {
    "AAPL": "AAPL",    # Apple Inc.
    "MSFT": "MSFT",    # Microsoft Corporation
    "GOOGL": "GOOGL",  # Alphabet Inc. (Google)
    "NVDA": "NVDA",    # NVIDIA Corporation
    "AMZN": "AMZN",    # Amazon.com Inc.
}

# Load unified configuration
from bot.config import load_config
app_config = load_config()

# Trading Configuration - prefer env vars, then unified config
LOOP_INTERVAL = int(os.getenv("STOCK_LOOP_INTERVAL", str(app_config.trading.loop_interval)))  # seconds
INITIAL_CAPITAL = float(os.getenv("STOCK_INITIAL_CAPITAL", str(app_config.trading.initial_capital)))
REBALANCE_THRESHOLD = float(os.getenv("STOCK_REBALANCE_THRESHOLD", str(app_config.trading.rebalance_threshold)))
SYMBOL_FETCH_DELAY = app_config.trading.symbol_fetch_delay  # Delay between fetching each symbol

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Deep Learning Configuration
# DL predictions are disabled for stocks (no trained models yet)
USE_DEEP_LEARNING = False
DL_MODEL_SELECTION = "regime_based"


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

    alert = Alert(
        alert_type=alert_type,
        level=AlertLevel.ALERT,
        title=f"{action} Signal: {symbol}",
        message=f"Executed {action} order for {symbol}\n\n"
                f"Quantity: {quantity:.4f} shares\n"
                f"Price: ${price:,.2f}\n"
                f"Value: ${quantity * price:,.2f}\n"
                f"Regime: {regime}\n"
                f"Confidence: {confidence:.1%}",
        data={
            "symbol": symbol,
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

    alert = Alert(
        alert_type=AlertType.SIGNAL_GENERATED,
        level=AlertLevel.INFO,
        title=f"{action} Signal: {symbol}",
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

    symbols_list = ", ".join(SYMBOL_MAP.keys())

    alert = Alert(
        alert_type=AlertType.SYSTEM,
        level=AlertLevel.INFO,
        title="Stock Trading Bot Started",
        message=f"Stock paper trading session started\n\n"
                f"Initial Capital: ${capital:,.2f}\n"
                f"Stocks: {symbols_list}\n"
                f"Strategy: Risk Parity + ML Regime Detection\n"
                f"Rebalance Threshold: {REBALANCE_THRESHOLD:.0%}",
        data={"capital": capital, "loop_interval": LOOP_INTERVAL},
    )

    manager.send_alert(alert)


def send_shutdown_alert(manager: NotificationManager, portfolio_value: float, pnl: float):
    """Send shutdown notification."""
    if not manager.has_channels():
        return

    pnl_status = "PROFIT" if pnl >= 0 else "LOSS"

    alert = Alert(
        alert_type=AlertType.SYSTEM,
        level=AlertLevel.WARNING,
        title="Stock Trading Bot Stopped",
        message=f"Stock trading session ended\n\n"
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
        memory_ttl=60.0,  # 1 minute
        disk_ttl=300.0,   # 5 minutes
    )
    logger.info("Rate limiter and cache initialized")
except ImportError:
    _rate_limiter = None
    _cache = None
    logger.warning("Rate limiter not available, using simple delays")


def fetch_market_data(symbol: str, period: str = "60d", interval: str = "1h"):
    """
    Fetch real market data from Yahoo Finance with rate limiting and caching.

    Args:
        symbol: Internal symbol name (e.g., "AAPL")
        period: Data period (e.g., "60d" for 60 days)
        interval: Data interval (e.g., "1h" for hourly)

    Returns:
        DataFrame with OHLCV data or None if fetch failed
    """
    yf_symbol = SYMBOL_MAP.get(symbol)
    if not yf_symbol:
        logger.warning(f"Unknown symbol: {symbol}")
        return None

    # Check cache first
    cache_key = f"stock:{yf_symbol}:{period}:{interval}"
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
        # Fallback to simple delay if no rate limiter
        time.sleep(5)

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)

        if _rate_limiter:
            _rate_limiter.report_success()

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        df.columns = [c.lower() for c in df.columns]
        result = df[["open", "high", "low", "close", "volume"]]

        # Cache the result
        if _cache and not result.empty:
            # Convert to dict for JSON serialization
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
    """Main trading loop for stock trading."""
    logger.info("=" * 60)
    logger.info("STARTING STOCK TRADING BOT")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Loop Interval: {LOOP_INTERVAL}s")
    logger.info(f"Rebalance Threshold: {REBALANCE_THRESHOLD:.0%}")
    logger.info(f"Telegram Alerts: {'Enabled' if TELEGRAM_ENABLED else 'Disabled'}")
    logger.info(f"Deep Learning: Disabled (no trained models for stocks)")
    logger.info(f"Stocks: {', '.join(SYMBOL_MAP.keys())}")
    logger.info("=" * 60)

    # Setup notifications
    notifier = setup_notifications()

    # Configure portfolio for stocks
    # Using conservative weight limits for stock portfolio
    config = MultiAssetConfig(
        assets=[
            AssetConfig(symbol="AAPL", asset_type="stock", max_weight=0.30, min_weight=0.10),
            AssetConfig(symbol="MSFT", asset_type="stock", max_weight=0.30, min_weight=0.10),
            AssetConfig(symbol="GOOGL", asset_type="stock", max_weight=0.25, min_weight=0.10),
            AssetConfig(symbol="NVDA", asset_type="stock", max_weight=0.25, min_weight=0.05),
            AssetConfig(symbol="AMZN", asset_type="stock", max_weight=0.25, min_weight=0.10),
        ],
        optimization_method=OptimizationMethod.RISK_PARITY,
        rebalance_threshold=REBALANCE_THRESHOLD,
        rebalance_frequency="daily",
        total_capital=INITIAL_CAPITAL,
        use_correlation_filter=True,  # Stocks can be highly correlated
        max_correlation=0.90,  # Filter highly correlated stocks
    )

    # Initialize engine with stock-specific data directory
    engine = MultiAssetTradingEngine(
        config=config,
        data_dir=str(DATA_DIR),
    )

    # Send startup alert
    send_startup_alert(notifier, INITIAL_CAPITAL)

    iteration = 0
    last_prices = {}
    last_signals = {}  # Track last signals to avoid duplicate alerts

    while True:
        iteration += 1
        loop_start = time.time()

        try:
            logger.info(f"--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # Fetch market data for all stocks with delay between symbols
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

                arrow = "+" if change_pct > 0.1 else "-" if change_pct < -0.1 else "="
                logger.info(f"  {symbol}: ${current_price:,.2f} {arrow} ({change_pct:+.2f}%)")
                last_prices[symbol] = current_price

            # Analyze portfolio
            decision = engine.analyze_portfolio(market_data)

            # Log ML decisions and send signal alerts for new signals
            for ad in decision.asset_decisions:
                signal = "LONG" if ad.action == "LONG" else "SHORT" if ad.action == "SHORT" else "FLAT"
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
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "market_type": "stock",
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
                        # No DL prediction for stocks yet
                        "dl_prediction": None,
                    }
                    for ad in decision.asset_decisions
                },
                "deep_learning_enabled": False,
                "dl_model_selection": None,
                "stocks": list(SYMBOL_MAP.keys()),
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state_data, f, indent=2)

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
    logger.info("STOCK TRADING SESSION ENDED")
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
            logger.info(f"  {symbol}: {pos['quantity']:.4f} shares @ ${pos['price']:,.2f} = ${pos['value']:,.2f}")

        # Send shutdown alert
        send_shutdown_alert(notifier, portfolio_value, pnl)

    except Exception as e:
        logger.error(f"Error getting final status: {e}")


if __name__ == "__main__":
    run_trading_loop()
