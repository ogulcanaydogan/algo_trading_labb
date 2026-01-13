#!/usr/bin/env python3
"""
Multi-Market Trading Orchestrator.

Manages multiple trading bots (crypto, commodity, stock) as subprocesses.
Provides unified monitoring, health checks, and consolidated Telegram alerts.

Features:
- Launch and manage trading bots as subprocesses
- Monitor health via log files and state files
- Unified command-line interface
- Support starting/stopping individual markets
- Consolidated Telegram alerts (daily summary)
- Graceful shutdown handling (SIGINT, SIGTERM)

Usage:
    python run_multi_market.py --all                     # Start all markets
    python run_multi_market.py --markets crypto,stock    # Start specific markets
    python run_multi_market.py --dry-run                 # Test without starting bots
    python run_multi_market.py --status                  # Check status only
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load unified configuration
from bot.config import load_config, get_data_dir

app_config = load_config()

# Configuration from unified config system
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = get_data_dir()
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Build market descriptions from config
crypto_desc = ", ".join(s.split("/")[0] for s in app_config.crypto.symbols[:4])
commodity_desc = ", ".join(s.split("/")[0] for s in app_config.commodities.symbols[:4])
stock_desc = ", ".join(app_config.stocks.symbols[:5])

# Scripts directory (same directory as this file)
SCRIPTS_DIR = Path(__file__).parent.resolve()

# Market configurations
MARKET_CONFIGS = {
    "crypto": {
        "script": str(SCRIPTS_DIR / "run_live_paper_trading.py"),
        "state_file": DATA_DIR / "live_paper_trading" / "state.json",
        "log_file": LOG_DIR / "paper_trading.log",
        "display_name": "Crypto",
        "description": crypto_desc,
        "enabled": app_config.crypto.enabled,
    },
    "commodity": {
        "script": str(SCRIPTS_DIR / "run_commodity_trading.py"),
        "state_file": DATA_DIR / "commodity_trading" / "state.json",
        "log_file": LOG_DIR / "commodity_trading.log",
        "display_name": "Commodity",
        "description": commodity_desc,
        "enabled": app_config.commodities.enabled,
    },
    "stock": {
        "script": str(SCRIPTS_DIR / "run_stock_trading.py"),
        "state_file": DATA_DIR / "stock_trading" / "state.json",
        "log_file": LOG_DIR / "stock_trading.log",
        "display_name": "Stock",
        "description": stock_desc,
        "enabled": app_config.stocks.enabled,
    },
}

# Orchestrator settings from unified config
HEALTH_CHECK_INTERVAL = app_config.orchestrator.health_check_interval
DAILY_SUMMARY_HOUR = app_config.notifications.daily_summary_hour
MAX_RESTART_ATTEMPTS = app_config.orchestrator.max_restart_attempts
RESTART_COOLDOWN = app_config.orchestrator.restart_cooldown

# Telegram configuration (secrets still from env vars)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = app_config.notifications.telegram_enabled and bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Setup logging using config
log_level = getattr(logging, app_config.general.log_level, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | [orchestrator] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "multi_market.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("multi-market-orchestrator")


@dataclass
class MarketStatus:
    """Status of a single market bot."""
    market: str
    is_running: bool = False
    pid: Optional[int] = None
    total_value: float = 0.0
    initial_capital: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    positions_count: int = 0
    last_update: Optional[datetime] = None
    health_status: str = "unknown"
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ProcessInfo:
    """Information about a running subprocess."""
    market: str
    process: subprocess.Popen
    started_at: datetime
    restart_count: int = 0
    last_restart: Optional[datetime] = None


class TelegramNotifier:
    """Simple Telegram notification sender."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message: str) -> bool:
        """Send a message via Telegram."""
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = json.dumps({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200

        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")
            return False


class MultiMarketOrchestrator:
    """
    Orchestrates multiple trading bots.

    Manages subprocess lifecycle, health monitoring, and consolidated reporting.
    """

    def __init__(
        self,
        markets: List[str],
        dry_run: bool = False,
        auto_restart: bool = True,
    ):
        self.markets = markets
        self.dry_run = dry_run
        self.auto_restart = auto_restart

        self.processes: Dict[str, ProcessInfo] = {}
        self.market_status: Dict[str, MarketStatus] = {}
        self._shutdown_requested = False
        self._lock = threading.Lock()

        # Initialize market status
        for market in markets:
            self.market_status[market] = MarketStatus(market=market)

        # Setup Telegram notifier
        self.notifier: Optional[TelegramNotifier] = None
        if TELEGRAM_ENABLED:
            self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

        # Track last daily summary
        self._last_daily_summary: Optional[datetime] = None

        # Alert throttling - prevent spam
        self._last_alerts: Dict[str, datetime] = {}
        self._alert_throttle_seconds = 300  # 5 minutes between same alerts

        # Register signal handlers
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Register handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.shutdown)

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def start(self):
        """Start all configured market bots."""
        logger.info("=" * 70)
        logger.info("MULTI-MARKET TRADING ORCHESTRATOR")
        logger.info("=" * 70)
        logger.info(f"Markets: {', '.join(self.markets)}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info(f"Auto Restart: {self.auto_restart}")
        logger.info(f"Health Check Interval: {HEALTH_CHECK_INTERVAL}s")
        logger.info(f"Daily Summary Hour: {DAILY_SUMMARY_HOUR}:00")
        logger.info("=" * 70)

        if self.dry_run:
            logger.info("[DRY RUN] Would start the following bots:")
            for market in self.markets:
                config = MARKET_CONFIGS.get(market, {})
                logger.info(f"  - {market}: {config.get('script', 'unknown')}")
            logger.info("[DRY RUN] Checking current state files...")
            self._check_all_states()
            return

        # Send startup notification
        self._send_startup_alert()

        # Start all market bots
        for market in self.markets:
            self._start_market(market)

        # Run main monitoring loop
        self._monitoring_loop()

    def _start_market(self, market: str) -> bool:
        """Start a single market bot."""
        if market not in MARKET_CONFIGS:
            logger.error(f"Unknown market: {market}")
            return False

        config = MARKET_CONFIGS[market]
        script_path = BASE_DIR / config["script"]

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            self.market_status[market].error_message = f"Script not found: {script_path}"
            return False

        # Check if already running
        if market in self.processes:
            proc_info = self.processes[market]
            if proc_info.process.poll() is None:
                logger.warning(f"{market} is already running (PID: {proc_info.process.pid})")
                return True

        try:
            # Ensure data directory exists
            state_file = config["state_file"]
            state_file.parent.mkdir(parents=True, exist_ok=True)

            # Start subprocess with PYTHONPATH set to project root
            logger.info(f"Starting {market} bot: {script_path}")

            # Set environment with PYTHONPATH pointing to project root
            env = os.environ.copy()
            project_root = Path(__file__).parent.parent.parent.resolve()
            env["PYTHONPATH"] = str(project_root)

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Store process info
            with self._lock:
                restart_count = 0
                last_restart = None
                if market in self.processes:
                    restart_count = self.processes[market].restart_count
                    last_restart = datetime.now()

                self.processes[market] = ProcessInfo(
                    market=market,
                    process=process,
                    started_at=datetime.now(),
                    restart_count=restart_count,
                    last_restart=last_restart,
                )

                self.market_status[market].is_running = True
                self.market_status[market].pid = process.pid
                self.market_status[market].restart_count = restart_count
                self.market_status[market].last_restart = last_restart
                self.market_status[market].error_message = None

            logger.info(f"Started {market} bot (PID: {process.pid})")

            # Start output reader thread
            threading.Thread(
                target=self._read_process_output,
                args=(market, process),
                daemon=True,
            ).start()

            return True

        except Exception as e:
            logger.exception(f"Failed to start {market} bot: {e}")
            self.market_status[market].error_message = str(e)
            return False

    def _read_process_output(self, market: str, process: subprocess.Popen):
        """Read and log subprocess output."""
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                # Log with market prefix (remove newline)
                line = line.rstrip()
                if line:
                    logger.debug(f"[{market}] {line}")
        except Exception as e:
            logger.warning(f"Error reading {market} output: {e}")

    def stop_market(self, market: str, timeout: int = 30) -> bool:
        """Stop a single market bot gracefully."""
        if market not in self.processes:
            logger.warning(f"{market} is not running")
            return True

        proc_info = self.processes[market]
        process = proc_info.process

        if process.poll() is not None:
            logger.info(f"{market} has already stopped")
            with self._lock:
                del self.processes[market]
                self.market_status[market].is_running = False
                self.market_status[market].pid = None
            return True

        logger.info(f"Stopping {market} bot (PID: {process.pid})...")

        try:
            # Send SIGINT for graceful shutdown
            process.send_signal(signal.SIGINT)

            # Wait for process to terminate
            try:
                process.wait(timeout=timeout)
                logger.info(f"{market} bot stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"{market} did not stop gracefully, sending SIGTERM...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.error(f"{market} did not terminate, killing...")
                    process.kill()
                    process.wait()

            with self._lock:
                del self.processes[market]
                self.market_status[market].is_running = False
                self.market_status[market].pid = None

            return True

        except Exception as e:
            logger.exception(f"Error stopping {market}: {e}")
            return False

    def _restart_market(self, market: str) -> bool:
        """Restart a market bot with cooldown and max attempts."""
        status = self.market_status[market]

        # Check restart cooldown
        if status.last_restart:
            cooldown_elapsed = (datetime.now() - status.last_restart).total_seconds()
            if cooldown_elapsed < RESTART_COOLDOWN:
                remaining = RESTART_COOLDOWN - cooldown_elapsed
                logger.warning(
                    f"Cannot restart {market}: cooldown active ({remaining:.0f}s remaining)"
                )
                return False

        # Check max restart attempts
        if status.restart_count >= MAX_RESTART_ATTEMPTS:
            logger.error(
                f"Max restart attempts ({MAX_RESTART_ATTEMPTS}) reached for {market}. "
                "Manual intervention required."
            )
            config = MARKET_CONFIGS.get(market, {})
            display = config.get("display_name", market)
            self._send_alert(
                f"🚨 <b>CRITICAL: {display} Bot Failed</b>\n\n"
                f"Bot failed after {MAX_RESTART_ATTEMPTS} restart attempts.\n"
                f"<b>Manual intervention required.</b>\n\n"
                f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            )
            return False

        # Stop if running
        self.stop_market(market)

        # Increment restart count before starting
        with self._lock:
            self.market_status[market].restart_count += 1

        logger.info(
            f"Restarting {market} (attempt {status.restart_count + 1}/{MAX_RESTART_ATTEMPTS})..."
        )

        # Start market
        success = self._start_market(market)

        config = MARKET_CONFIGS.get(market, {})
        display = config.get("display_name", market)

        if success:
            self._send_alert(
                f"🔄 <b>{display} Bot Restarted</b>\n\n"
                f"✅ Restart successful\n"
                f"Attempt: {status.restart_count}/{MAX_RESTART_ATTEMPTS}\n\n"
                f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            )
        else:
            self._send_alert(
                f"⚠️ <b>{display} Bot Restart Failed</b>\n\n"
                f"❌ Failed to restart\n"
                f"Attempt: {status.restart_count}/{MAX_RESTART_ATTEMPTS}\n\n"
                f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            )

        return success

    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop...")

        while not self._shutdown_requested:
            try:
                # Check process health
                self._check_process_health()

                # Update market status from state files
                self._check_all_states()

                # Log combined status
                self._log_combined_status()

                # Check for daily summary
                self._check_daily_summary()

                # Sleep
                time.sleep(HEALTH_CHECK_INTERVAL)

            except Exception as e:
                logger.exception(f"Error in monitoring loop: {e}")
                time.sleep(10)

        # Shutdown requested
        logger.info("Shutdown requested, stopping all bots...")
        self.shutdown()

    def _check_process_health(self):
        """Check health of all running processes."""
        for market in list(self.processes.keys()):
            proc_info = self.processes[market]
            process = proc_info.process

            # Check if process is still running
            poll_result = process.poll()
            if poll_result is not None:
                logger.warning(
                    f"{market} bot exited unexpectedly (exit code: {poll_result})"
                )

                with self._lock:
                    self.market_status[market].is_running = False
                    self.market_status[market].pid = None
                    self.market_status[market].health_status = "crashed"
                    self.market_status[market].error_message = (
                        f"Process exited with code {poll_result}"
                    )

                # Attempt restart if enabled
                if self.auto_restart and not self._shutdown_requested:
                    self._restart_market(market)

    def _check_all_states(self):
        """Check state files for all markets."""
        for market in self.markets:
            self._update_market_status(market)

    def _update_market_status(self, market: str):
        """Update market status from state file."""
        config = MARKET_CONFIGS.get(market, {})
        state_file = config.get("state_file")

        if not state_file or not state_file.exists():
            self.market_status[market].health_status = "no_state_file"
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            with self._lock:
                status = self.market_status[market]
                status.total_value = state.get("total_value", 0)
                status.initial_capital = state.get("initial_capital", 0)
                status.pnl = state.get("pnl", 0)
                status.pnl_pct = state.get("pnl_pct", 0)
                status.positions_count = state.get("positions_count", 0)

                # Parse timestamp
                timestamp_str = state.get("timestamp")
                if timestamp_str:
                    status.last_update = datetime.fromisoformat(timestamp_str)

                # Determine health status
                if status.last_update:
                    age = (datetime.now() - status.last_update).total_seconds()
                    if age < 120:  # Updated within 2 minutes
                        status.health_status = "healthy"
                    elif age < 300:  # Updated within 5 minutes
                        status.health_status = "stale"
                    else:
                        status.health_status = "unresponsive"
                else:
                    status.health_status = "unknown"

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {market} state file: {e}")
            self.market_status[market].health_status = "invalid_state"
        except Exception as e:
            logger.warning(f"Error reading {market} state: {e}")
            self.market_status[market].health_status = "error"

    def _log_combined_status(self):
        """Log combined portfolio status."""
        combined_value = 0.0
        combined_initial = 0.0
        total_positions = 0

        lines = []
        lines.append("-" * 70)
        lines.append(f"PORTFOLIO STATUS | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-" * 70)

        for market in self.markets:
            status = self.market_status[market]
            config = MARKET_CONFIGS.get(market, {})

            # Running indicator
            running = "RUN" if status.is_running else "OFF"
            health = status.health_status[:3].upper()

            # P&L indicator
            pnl_sign = "+" if status.pnl >= 0 else ""

            line = (
                f"  {config.get('display_name', market):10} | "
                f"{running:3} | {health:3} | "
                f"Value: ${status.total_value:>10,.2f} | "
                f"P&L: {pnl_sign}{status.pnl_pct:>6.2f}% | "
                f"Pos: {status.positions_count}"
            )
            lines.append(line)

            combined_value += status.total_value
            combined_initial += status.initial_capital
            total_positions += status.positions_count

        lines.append("-" * 70)

        # Combined totals
        combined_pnl = combined_value - combined_initial
        combined_pnl_pct = (
            (combined_pnl / combined_initial * 100) if combined_initial > 0 else 0
        )
        pnl_sign = "+" if combined_pnl >= 0 else ""

        lines.append(
            f"  {'COMBINED':10} |     |     | "
            f"Value: ${combined_value:>10,.2f} | "
            f"P&L: {pnl_sign}{combined_pnl_pct:>6.2f}% | "
            f"Pos: {total_positions}"
        )
        lines.append("-" * 70)

        # Log as single block
        for line in lines:
            logger.info(line)

    def _check_daily_summary(self):
        """Check if it's time to send daily summary."""
        now = datetime.now()

        # Check if it's the right hour
        if now.hour != DAILY_SUMMARY_HOUR:
            return

        # Check if we already sent today
        if self._last_daily_summary:
            if self._last_daily_summary.date() == now.date():
                return

        # Send daily summary
        self._send_daily_summary()
        self._last_daily_summary = now

    def _send_daily_summary(self):
        """Send consolidated daily summary via Telegram with rich HTML formatting."""
        logger.info("Sending daily summary...")

        combined_value = 0.0
        combined_initial = 0.0
        combined_trades = 0
        combined_wins = 0

        market_lines = []
        for market in self.markets:
            status = self.market_status[market]
            config = MARKET_CONFIGS.get(market, {})
            display = config.get("display_name", market)

            # Market emoji
            market_emoji = {"Crypto": "🪙", "Commodity": "🛢️", "Stock": "📈"}.get(display, "📊")

            # Health emoji
            health_emoji = {
                "healthy": "✅",
                "stale": "⚠️",
                "unresponsive": "❌",
            }.get(status.health_status, "❓")

            # P&L formatting
            pnl_str = f"+${status.pnl:,.2f}" if status.pnl >= 0 else f"-${abs(status.pnl):,.2f}"
            pnl_pct_str = f"+{status.pnl_pct:.2f}%" if status.pnl_pct >= 0 else f"{status.pnl_pct:.2f}%"

            market_lines.append(
                f"  {market_emoji} <b>{display}</b> {health_emoji}\n"
                f"     Value: ${status.total_value:,.2f} | P&L: {pnl_str} ({pnl_pct_str})"
            )

            combined_value += status.total_value
            combined_initial += status.initial_capital
            combined_trades += getattr(status, "total_trades", 0)
            combined_wins += getattr(status, "wins", 0)

        # Combined totals
        combined_pnl = combined_value - combined_initial
        combined_pnl_pct = (
            (combined_pnl / combined_initial * 100) if combined_initial > 0 else 0
        )

        # Header emoji based on performance
        if combined_pnl_pct >= 3:
            header_emoji = "🚀"
            mood = "Excellent"
        elif combined_pnl_pct >= 1:
            header_emoji = "📈"
            mood = "Good"
        elif combined_pnl_pct >= 0:
            header_emoji = "➖"
            mood = "Flat"
        elif combined_pnl_pct >= -2:
            header_emoji = "📉"
            mood = "Challenging"
        else:
            header_emoji = "⚠️"
            mood = "Difficult"

        market_summary = "\n".join(market_lines)

        # Performance bar (scale from -5% to +5%)
        perf_scaled = min(10, max(0, int((combined_pnl_pct + 5) / 1)))
        perf_bar = "🟢" * perf_scaled + "⚪" * (10 - perf_scaled)

        message = f"""
{header_emoji} <b>Daily Summary: {mood} Day</b>
<i>{datetime.now().strftime('%Y-%m-%d')}</i>

<b>Markets:</b>
{market_summary}

<b>Combined Portfolio:</b>
• Total Value: <b>${combined_value:,.2f}</b>
• Daily P&L: <b>${combined_pnl:+,.2f}</b> ({combined_pnl_pct:+.2f}%)

{perf_bar}

<i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        self._send_alert(message.strip())

    def _send_startup_alert(self):
        """Send startup notification with rich HTML formatting."""
        market_items = []
        for m in self.markets:
            config = MARKET_CONFIGS.get(m, {})
            display = config.get("display_name", m)
            desc = config.get("description", "")
            emoji = {"Crypto": "🪙", "Commodity": "🛢️", "Stock": "📈"}.get(display, "📊")
            market_items.append(f"  {emoji} <b>{display}</b>: {desc}")

        market_list = "\n".join(market_items)

        message = f"""
🚀 <b>Multi-Market Orchestrator Started</b>

<b>Active Markets:</b>
{market_list}

<b>Settings:</b>
• Auto-Restart: {'✅ Enabled' if self.auto_restart else '❌ Disabled'}
• Health Check: Every {HEALTH_CHECK_INTERVAL}s
• Daily Summary: {DAILY_SUMMARY_HOUR}:00

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_alert(message.strip())

    def _send_shutdown_alert(self):
        """Send shutdown notification with rich HTML formatting."""
        # Get final combined status
        combined_value = sum(s.total_value for s in self.market_status.values())
        combined_initial = sum(s.initial_capital for s in self.market_status.values())
        combined_pnl = combined_value - combined_initial
        combined_pnl_pct = (
            (combined_pnl / combined_initial * 100) if combined_initial > 0 else 0
        )

        # P&L emoji
        if combined_pnl >= 0:
            pnl_emoji = "📈" if combined_pnl_pct >= 1 else "➖"
        else:
            pnl_emoji = "📉" if combined_pnl_pct <= -1 else "➖"

        # Build market summary
        market_lines = []
        for market in self.markets:
            status = self.market_status[market]
            config = MARKET_CONFIGS.get(market, {})
            display = config.get("display_name", market)
            emoji = {"Crypto": "🪙", "Commodity": "🛢️", "Stock": "📈"}.get(display, "📊")
            pnl_str = f"+${status.pnl:,.2f}" if status.pnl >= 0 else f"-${abs(status.pnl):,.2f}"
            market_lines.append(f"  {emoji} {display}: ${status.total_value:,.2f} ({pnl_str})")

        market_summary = "\n".join(market_lines)

        message = f"""
⏹️ <b>Multi-Market Orchestrator Stopped</b>

<b>Market Summary:</b>
{market_summary}

<b>Combined:</b>
• Value: <b>${combined_value:,.2f}</b>
• P&L: {pnl_emoji} <b>${combined_pnl:+,.2f}</b> ({combined_pnl_pct:+.2f}%)

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_alert(message.strip())

    def _send_alert(self, message: str, force: bool = False):
        """Send alert via Telegram with throttling to prevent spam."""
        # Create a key from the message (first 50 chars)
        alert_key = message[:50].strip()

        # Check throttling (unless forced)
        if not force and alert_key in self._last_alerts:
            elapsed = (datetime.now() - self._last_alerts[alert_key]).total_seconds()
            if elapsed < self._alert_throttle_seconds:
                logger.debug(f"Alert throttled ({elapsed:.0f}s < {self._alert_throttle_seconds}s): {alert_key}")
                return

        # Update last alert time
        self._last_alerts[alert_key] = datetime.now()

        # Clean old alerts (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self._last_alerts = {k: v for k, v in self._last_alerts.items() if v > cutoff}

        # Send alert
        if self.notifier:
            self.notifier.send_message(message)
        logger.info(f"[ALERT] {message.replace(chr(10), ' | ')}")

    def shutdown(self):
        """Gracefully shutdown all bots."""
        if self._shutdown_requested and not self.processes:
            return  # Already shutdown

        logger.info("=" * 70)
        logger.info("SHUTTING DOWN MULTI-MARKET ORCHESTRATOR")
        logger.info("=" * 70)

        # Update final status
        self._check_all_states()

        # Send shutdown alert
        if not self.dry_run:
            self._send_shutdown_alert()

        # Stop all processes
        for market in list(self.processes.keys()):
            self.stop_market(market)

        logger.info("All bots stopped. Goodbye!")

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        # Update states first
        self._check_all_states()

        combined_value = sum(s.total_value for s in self.market_status.values())
        combined_initial = sum(s.initial_capital for s in self.market_status.values())

        return {
            "timestamp": datetime.now().isoformat(),
            "markets": {
                market: {
                    "is_running": status.is_running,
                    "pid": status.pid,
                    "health_status": status.health_status,
                    "total_value": status.total_value,
                    "initial_capital": status.initial_capital,
                    "pnl": status.pnl,
                    "pnl_pct": status.pnl_pct,
                    "positions_count": status.positions_count,
                    "last_update": (
                        status.last_update.isoformat() if status.last_update else None
                    ),
                    "restart_count": status.restart_count,
                    "error_message": status.error_message,
                }
                for market, status in self.market_status.items()
            },
            "combined": {
                "total_value": combined_value,
                "initial_capital": combined_initial,
                "pnl": combined_value - combined_initial,
                "pnl_pct": (
                    ((combined_value - combined_initial) / combined_initial * 100)
                    if combined_initial > 0
                    else 0
                ),
            },
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Market Trading Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_multi_market.py --all
  python run_multi_market.py --markets crypto,commodity
  python run_multi_market.py --markets stock --dry-run
  python run_multi_market.py --status
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Start all markets (crypto, commodity, stock)",
    )

    parser.add_argument(
        "--markets",
        type=str,
        help="Comma-separated list of markets to start (crypto,commodity,stock)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: check configuration without starting bots",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Check and display current status only",
    )

    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Disable automatic restart of crashed bots",
    )

    return parser.parse_args()


def show_status():
    """Display current status from state files."""
    print("=" * 70)
    print("MULTI-MARKET STATUS CHECK")
    print("=" * 70)

    combined_value = 0.0
    combined_initial = 0.0

    for market, config in MARKET_CONFIGS.items():
        state_file = config["state_file"]
        print(f"\n{config['display_name']} ({market}):")
        print(f"  Script: {config['script']}")
        print(f"  Assets: {config['description']}")

        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)

                total_value = state.get("total_value", 0)
                initial_capital = state.get("initial_capital", 0)
                pnl = state.get("pnl", 0)
                pnl_pct = state.get("pnl_pct", 0)
                positions = state.get("positions_count", 0)
                timestamp = state.get("timestamp", "unknown")

                pnl_sign = "+" if pnl >= 0 else ""

                print(f"  State File: {state_file}")
                print(f"  Last Update: {timestamp}")
                print(f"  Total Value: ${total_value:,.2f}")
                print(f"  Initial Capital: ${initial_capital:,.2f}")
                print(f"  P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)")
                print(f"  Positions: {positions}")

                combined_value += total_value
                combined_initial += initial_capital

            except Exception as e:
                print(f"  State File: {state_file} (ERROR: {e})")
        else:
            print(f"  State File: NOT FOUND ({state_file})")

    print("\n" + "=" * 70)
    combined_pnl = combined_value - combined_initial
    combined_pnl_pct = (
        (combined_pnl / combined_initial * 100) if combined_initial > 0 else 0
    )
    pnl_sign = "+" if combined_pnl >= 0 else ""

    print("COMBINED PORTFOLIO:")
    print(f"  Total Value: ${combined_value:,.2f}")
    print(f"  Initial Capital: ${combined_initial:,.2f}")
    print(f"  P&L: {pnl_sign}${combined_pnl:,.2f} ({pnl_sign}{combined_pnl_pct:.2f}%)")
    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()

    # Status check only
    if args.status:
        show_status()
        return

    # Determine which markets to run
    markets = []

    if args.all:
        markets = list(MARKET_CONFIGS.keys())
    elif args.markets:
        markets = [m.strip().lower() for m in args.markets.split(",")]
        # Validate markets
        invalid = [m for m in markets if m not in MARKET_CONFIGS]
        if invalid:
            print(f"Error: Unknown market(s): {', '.join(invalid)}")
            print(f"Valid markets: {', '.join(MARKET_CONFIGS.keys())}")
            sys.exit(1)
    else:
        print("Error: Specify --all or --markets")
        print("Use --help for usage information")
        sys.exit(1)

    # Create and start orchestrator
    orchestrator = MultiMarketOrchestrator(
        markets=markets,
        dry_run=args.dry_run,
        auto_restart=not args.no_auto_restart,
    )

    try:
        orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
