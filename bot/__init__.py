"""Algo Trading Lab bot package."""

import os

# Initialize logging once at package import
# This prevents duplicate logging.basicConfig calls across modules
_logging_initialized = False

def _init_logging():
    """Initialize logging configuration once."""
    global _logging_initialized
    if _logging_initialized:
        return

    try:
        from bot.core.logging_config import setup_logging

        log_level = os.getenv("LOG_LEVEL", "INFO")
        json_format = os.getenv("LOG_FORMAT", "").lower() == "json"
        log_file = os.getenv("LOG_FILE")

        setup_logging(
            level=log_level,
            json_format=json_format,
            log_file=log_file,
            include_trade_logger=True,
        )
        _logging_initialized = True
    except Exception:
        # Fallback to basic logging if setup fails
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        _logging_initialized = True

# Auto-initialize logging when bot package is imported
_init_logging()

from .bot import BotConfig, run_loop
from .portfolio import PortfolioAssetConfig, PortfolioConfig, PortfolioRunner

__all__ = [
    "BotConfig",
    "PortfolioAssetConfig",
    "PortfolioConfig",
    "PortfolioRunner",
    "run_loop",
]
