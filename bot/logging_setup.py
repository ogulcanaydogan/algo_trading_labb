"""
Structured logging configuration for the trading system.
Supports JSON logging, file rotation, and multiple handlers.
"""

import logging
import logging.config
import json
from pathlib import Path
from datetime import datetime

# Ensure logs directory exists
LOGS_DIR = Path("data/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing and indexing."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry)

# Logging configuration dictionary
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s | %(levelname)-8s | [%(name)s:%(funcName)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': JSONFormatter,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'trading.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
        },
        'json_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': str(LOGS_DIR / 'trading.jsonl'),
            'maxBytes': 52428800,  # 50MB
            'backupCount': 5,
        },
        'api_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'api.log'),
            'maxBytes': 10485760,
            'backupCount': 5,
        },
        'bot_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'bot.log'),
            'maxBytes': 10485760,
            'backupCount': 5,
        },
        'errors_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'errors.log'),
            'maxBytes': 10485760,
            'backupCount': 10,
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file', 'json_file', 'errors_file'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'api': {
            'handlers': ['console', 'api_file', 'json_file', 'errors_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'bot': {
            'handlers': ['console', 'bot_file', 'json_file', 'errors_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'urllib3': {
            'handlers': ['errors_file'],
            'level': 'WARNING',
            'propagate': False,
        },
        'requests': {
            'handlers': ['errors_file'],
            'level': 'WARNING',
            'propagate': False,
        },
    }
}

def setup_logging():
    """Initialize structured logging."""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Structured logging initialized")
    return logger

def get_logger(name):
    """Get a logger instance with optional extra fields."""
    return logging.getLogger(name)

def log_with_context(logger, level, message, **context):
    """Log with additional context fields."""
    extra_data = {"extra": context} if context else {}
    getattr(logger, level.lower())(message, extra=context)
