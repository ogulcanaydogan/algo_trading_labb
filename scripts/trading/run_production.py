#!/usr/bin/env python3
"""
Production Trading Bot Runner

Features:
- Health monitoring with circuit breakers
- Auto-recovery on failures
- Telegram notifications
- Graceful shutdown handling
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot.production_engine import ProductionEngine, run_production_bot
from bot.unified_engine import EngineConfig
from bot.trading_mode import TradingMode


def setup_logging() -> None:
    """Configure production logging."""
    # Try user home directory first, then project directory
    import os
    home_log_dir = Path.home() / ".algo_trading" / "logs"
    project_log_dir = project_root / "data" / "logs"

    # Use home directory if project logs not writable
    try:
        project_log_dir.mkdir(parents=True, exist_ok=True)
        test_file = project_log_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        log_dir = project_log_dir
    except (PermissionError, OSError):
        home_log_dir.mkdir(parents=True, exist_ok=True)
        log_dir = home_log_dir

    log_file = log_dir / "production_trading.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger(__name__).info(f"Logging to: {log_file}")

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def load_config() -> EngineConfig:
    """Load configuration from environment and config files."""
    import json

    # Try to load from config file
    config_file = project_root / "data" / "production_config.json"

    if config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
    else:
        config_data = {}

    # Default configuration
    config = EngineConfig(
        initial_mode=TradingMode(
            config_data.get("mode", os.getenv("TRADING_MODE", "paper_live_data"))
        ),
        initial_capital=float(
            config_data.get("capital", os.getenv("INITIAL_CAPITAL", "10000"))
        ),
        symbols=config_data.get(
            "symbols",
            os.getenv("TRADING_SYMBOLS", "BTC/USDT,ETH/USDT").split(","),
        ),
        loop_interval_seconds=int(
            config_data.get("interval", os.getenv("LOOP_INTERVAL", "300"))
        ),
        risk_per_trade_pct=float(
            config_data.get("risk_per_trade", os.getenv("RISK_PER_TRADE", "0.01"))
        ),
        stop_loss_pct=float(
            config_data.get("stop_loss", os.getenv("STOP_LOSS_PCT", "0.02"))
        ),
        take_profit_pct=float(
            config_data.get("take_profit", os.getenv("TAKE_PROFIT_PCT", "0.04"))
        ),
        use_ml_signals=config_data.get(
            "use_ml", os.getenv("USE_ML_SIGNALS", "true").lower() == "true"
        ),
        ml_confidence_threshold=float(
            config_data.get("ml_confidence", os.getenv("ML_CONFIDENCE", "0.55"))
        ),
        data_dir=Path(
            config_data.get("data_dir", os.getenv("DATA_DIR", "data/production"))
        ),
    )

    return config


async def main() -> None:
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PRODUCTION TRADING BOT")
    logger.info("=" * 60)

    config = load_config()

    logger.info(f"Mode: {config.initial_mode.value}")
    logger.info(f"Capital: ${config.initial_capital:,.2f}")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Loop interval: {config.loop_interval_seconds}s")
    logger.info(f"ML signals: {config.use_ml_signals}")

    # Check for required environment variables
    required_vars = []
    if config.initial_mode.is_live:
        required_vars = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)

    # Telegram notification check
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if telegram_token and telegram_chat_id:
        logger.info("Telegram notifications: ENABLED")
    else:
        logger.warning("Telegram notifications: DISABLED (missing credentials)")

    logger.info("=" * 60)

    try:
        await run_production_bot(config)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
