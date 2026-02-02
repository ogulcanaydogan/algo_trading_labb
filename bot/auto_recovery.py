"""
Auto-Recovery Service for Production Trading

Automatically detects failures and recovers components:
- Restarts failed processes
- Reconnects dropped connections
- Clears stale state
- Escalates when recovery fails
"""

import asyncio
import inspect
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import os
import sys

logger = logging.getLogger(__name__)


def _resolve_venv_python() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if os.name == "nt":
        candidate = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _is_engine_running_windows() -> bool:
    # Windows-only guard using heartbeat/pidfile + cmdline validation.
    try:
        import psutil
    except ImportError:
        return False

    repo_root = Path(__file__).resolve().parent.parent
    heartbeat_path = repo_root / "data" / "rl" / "paper_live_heartbeat.json"
    pidfile_path = repo_root / "logs" / "paper_live.pid"
    pid = None

    if heartbeat_path.exists():
        try:
            with heartbeat_path.open("r", encoding="utf-8") as f:
                heartbeat = json.load(f)
            pid = heartbeat.get("pid")
        except (OSError, json.JSONDecodeError):
            pid = None

    if not pid and pidfile_path.exists():
        try:
            pid = int(pidfile_path.read_text(encoding="utf-8").strip().splitlines()[0])
        except (OSError, ValueError, IndexError):
            pid = None

    if not pid:
        return False

    try:
        proc = psutil.Process(int(pid))
        cmdline = " ".join(proc.cmdline())
        return "run_unified_trading.py" in cmdline
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


class RecoveryAction(Enum):
    """Types of recovery actions."""

    RESTART_PROCESS = "restart_process"
    RECONNECT = "reconnect"
    CLEAR_CACHE = "clear_cache"
    RESET_STATE = "reset_state"
    FAILOVER = "failover"
    ESCALATE = "escalate"


class RecoveryResult(Enum):
    """Result of recovery attempt."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    component: str
    action: RecoveryAction
    result: RecoveryResult
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class RecoveryConfig:
    """Configuration for auto-recovery."""

    max_attempts: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 300.0
    cooldown_period: float = 60.0
    escalation_threshold: int = 5


class AutoRecovery:
    """
    Automatic recovery service for production trading.

    Features:
    - Exponential backoff for retries
    - Cooldown to prevent recovery storms
    - Escalation when recovery fails repeatedly
    - Recovery history tracking
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        notification_callback: Optional[Callable] = None,
        escalation_callback: Optional[Callable] = None,
    ):
        self.config = config or RecoveryConfig()
        self.notification_callback = notification_callback
        self.escalation_callback = escalation_callback

        self.recovery_history: list[RecoveryAttempt] = []
        self.attempt_counts: dict[str, int] = {}
        self.last_recovery: dict[str, float] = {}
        self.recovery_handlers: dict[str, Callable] = {}

        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default recovery handlers."""

        self.recovery_handlers["process"] = self._recover_process
        self.recovery_handlers["connection"] = self._recover_connection
        self.recovery_handlers["cache"] = self._recover_cache
        self.recovery_handlers["state"] = self._recover_state
        self.recovery_handlers["data_feed"] = self._recover_data_feed
        self.recovery_handlers["api"] = self._recover_api

    def register_handler(
        self,
        component_type: str,
        handler: Callable,
    ) -> None:
        """Register a custom recovery handler."""
        self.recovery_handlers[component_type] = handler

    async def recover(
        self,
        component: str,
        component_type: str,
        context: Optional[dict[str, Any]] = None,
    ) -> RecoveryResult:
        """
        Attempt to recover a failed component.

        Args:
            component: Component identifier
            component_type: Type of component (process, connection, etc.)
            context: Additional context for recovery

        Returns:
            RecoveryResult indicating success/failure
        """
        context = context or {}

        # Check cooldown
        if self._in_cooldown(component):
            logger.info(f"Component {component} in cooldown, skipping recovery")
            return RecoveryResult.SKIPPED

        # Check attempt count
        attempts = self.attempt_counts.get(component, 0)
        if attempts >= self.config.max_attempts:
            logger.warning(f"Max recovery attempts reached for {component}")
            await self._escalate(component, "Max recovery attempts exceeded")
            return RecoveryResult.FAILED

        # Calculate backoff
        backoff = min(
            self.config.backoff_base**attempts,
            self.config.backoff_max,
        )

        if attempts > 0:
            logger.info(f"Waiting {backoff:.1f}s before recovery attempt {attempts + 1}")
            await asyncio.sleep(backoff)

        # Get handler
        handler = self.recovery_handlers.get(component_type)
        if not handler:
            logger.error(f"No handler for component type: {component_type}")
            return RecoveryResult.FAILED

        # Attempt recovery
        start_time = time.time()
        try:
            logger.info(f"Attempting recovery for {component} (type: {component_type})")

            success = await handler(component, context)

            duration_ms = (time.time() - start_time) * 1000

            if success:
                result = RecoveryResult.SUCCESS
                self.attempt_counts[component] = 0
                logger.info(f"Recovery successful for {component} in {duration_ms:.0f}ms")

                if self.notification_callback:
                    await self._notify(
                        f"✅ Recovery successful: {component}",
                        {"duration_ms": duration_ms},
                    )
            else:
                result = RecoveryResult.PARTIAL
                self.attempt_counts[component] = attempts + 1
                logger.warning(f"Partial recovery for {component}")

            self._record_attempt(
                component,
                RecoveryAction.RESTART_PROCESS,
                result,
                duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Recovery failed for {component}: {e}")

            self.attempt_counts[component] = attempts + 1
            self._record_attempt(
                component,
                RecoveryAction.RESTART_PROCESS,
                RecoveryResult.FAILED,
                duration_ms,
                str(e),
            )

            # Check escalation threshold
            total_failures = sum(
                1 for r in self.recovery_history[-10:] if r.result == RecoveryResult.FAILED
            )

            if total_failures >= self.config.escalation_threshold:
                await self._escalate(component, f"Too many failures: {str(e)}")

            return RecoveryResult.FAILED

        finally:
            self.last_recovery[component] = time.time()

    def _in_cooldown(self, component: str) -> bool:
        """Check if component is in cooldown period."""
        last = self.last_recovery.get(component, 0)
        return (time.time() - last) < self.config.cooldown_period

    def _record_attempt(
        self,
        component: str,
        action: RecoveryAction,
        result: RecoveryResult,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a recovery attempt."""
        attempt = RecoveryAttempt(
            component=component,
            action=action,
            result=result,
            duration_ms=duration_ms,
            error=error,
        )
        self.recovery_history.append(attempt)

        # Keep only last 100 attempts
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]

    async def _notify(self, message: str, context: dict[str, Any]) -> None:
        """Send notification."""
        if self.notification_callback:
            try:
                if inspect.iscoroutinefunction(self.notification_callback):
                    await self.notification_callback(message, context)
                else:
                    self.notification_callback(message, context)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    async def _escalate(self, component: str, reason: str) -> None:
        """Escalate unrecoverable failure."""
        logger.critical(f"ESCALATION: {component} - {reason}")

        if self.escalation_callback:
            try:
                if inspect.iscoroutinefunction(self.escalation_callback):
                    await self.escalation_callback(component, reason)
                else:
                    self.escalation_callback(component, reason)
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

        if self.notification_callback:
            await self._notify(
                f"🚨 ESCALATION: {component} requires manual intervention",
                {"reason": reason, "component": component},
            )

    # ==================== Default Recovery Handlers ====================

    async def _recover_process(
        self,
        component: str,
        context: dict[str, Any],
    ) -> bool:
        """Recover a failed process."""
        script = context.get("script")
        if not script:
            logger.error(f"No script specified for process recovery: {component}")
            return False

        try:
            if os.name == "nt":
                if "run_unified_trading.py" in script and _is_engine_running_windows():
                    return True

                # Use venv python on Windows to avoid system-python mismatches.
                python_exe = _resolve_venv_python()
                log_file = context.get("log_file")
                stdout_target = subprocess.DEVNULL
                stderr_target = subprocess.DEVNULL
                if log_file:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    stdout_target = open(log_path, "a", encoding="utf-8")
                    stderr_target = stdout_target

                process = subprocess.Popen(
                    [str(python_exe), str(script)],
                    stdout=stdout_target,
                    stderr=stderr_target,
                )

                # Keep log handle open in parent to avoid Windows log stream issues.

                await asyncio.sleep(2)
                return process.poll() is None

            # Kill existing process
            subprocess.run(
                ["pkill", "-f", script],
                capture_output=True,
                timeout=5,
            )
            await asyncio.sleep(1)

            # Start new process
            log_file = context.get("log_file", "/dev/null")
            cmd = f"nohup python {script} > {log_file} 2>&1 &"

            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            await asyncio.sleep(2)

            # Verify process started
            result = subprocess.run(
                ["pgrep", "-f", script],
                capture_output=True,
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Process recovery failed: {e}")
            return False

    async def _recover_connection(
        self,
        component: str,
        context: dict[str, Any],
    ) -> bool:
        """Recover a dropped connection."""
        reconnect_func = context.get("reconnect_func")
        if not reconnect_func:
            logger.error(f"No reconnect function for: {component}")
            return False

        try:
            if inspect.iscoroutinefunction(reconnect_func):
                await reconnect_func()
            else:
                reconnect_func()
            return True
        except Exception as e:
            logger.error(f"Connection recovery failed: {e}")
            return False

    async def _recover_cache(
        self,
        component: str,
        context: dict[str, Any],
    ) -> bool:
        """Clear and rebuild cache."""
        cache_path = context.get("cache_path")
        rebuild_func = context.get("rebuild_func")

        try:
            # Clear cache file
            if cache_path:
                path = Path(cache_path)
                if path.exists():
                    path.unlink()
                    logger.info(f"Cleared cache: {cache_path}")

            # Rebuild if function provided
            if rebuild_func:
                if inspect.iscoroutinefunction(rebuild_func):
                    await rebuild_func()
                else:
                    rebuild_func()

            return True

        except Exception as e:
            logger.error(f"Cache recovery failed: {e}")
            return False

    async def _recover_state(
        self,
        component: str,
        context: dict[str, Any],
    ) -> bool:
        """Reset component state."""
        state_file = context.get("state_file")
        default_state = context.get("default_state", {})

        try:
            import json

            if state_file:
                path = Path(state_file)

                # Backup existing state (run in thread to avoid blocking)
                def _backup_and_write():
                    if path.exists():
                        backup = path.with_suffix(".backup")
                        path.rename(backup)
                        logger.info(f"Backed up state to: {backup}")

                    # Write default state
                    with open(path, "w") as f:
                        json.dump(default_state, f, indent=2, default=str)

                await asyncio.to_thread(_backup_and_write)
                logger.info(f"Reset state: {state_file}")

            return True

        except Exception as e:
            logger.error(f"State recovery failed: {e}")
            return False

    async def _recover_data_feed(
        self,
        component: str,
        context: dict[str, Any],
    ) -> bool:
        """Recover a data feed connection."""
        provider = context.get("provider", "unknown")

        try:
            # Try to import and reconnect
            if provider == "binance":
                from bot.data_fetcher import SmartDataFetcher

                fetcher = SmartDataFetcher()
                # Test connection
                data = await fetcher.fetch_with_retry("BTC-USD", "1h")
                return data is not None

            elif provider == "yahoo":
                import yfinance as yf

                ticker = yf.Ticker("SPY")
                info = ticker.info
                return info is not None

            else:
                logger.warning(f"Unknown provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Data feed recovery failed: {e}")
            return False

    async def _recover_api(
        self,
        component: str,
        context: dict[str, Any],
    ) -> bool:
        """Recover API service."""
        port = context.get("port", 8000)

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        return True

            # API not responding, try restart
            subprocess.run(
                ["pkill", "-f", "uvicorn"],
                capture_output=True,
            )
            await asyncio.sleep(2)

            subprocess.Popen(
                f"uvicorn api.api:app --host 0.0.0.0 --port {port} &",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            await asyncio.sleep(3)

            # Verify
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"API recovery failed: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get recovery statistics."""
        total = len(self.recovery_history)
        if total == 0:
            return {
                "total_attempts": 0,
                "success_rate": 0,
                "by_result": {},
                "by_component": {},
            }

        by_result = {}
        by_component: dict[str, dict] = {}

        for attempt in self.recovery_history:
            # Count by result
            result_name = attempt.result.value
            by_result[result_name] = by_result.get(result_name, 0) + 1

            # Count by component
            if attempt.component not in by_component:
                by_component[attempt.component] = {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                }

            by_component[attempt.component]["total"] += 1
            if attempt.result == RecoveryResult.SUCCESS:
                by_component[attempt.component]["success"] += 1
            elif attempt.result == RecoveryResult.FAILED:
                by_component[attempt.component]["failed"] += 1

        success_count = by_result.get("success", 0)

        return {
            "total_attempts": total,
            "success_rate": success_count / total if total > 0 else 0,
            "by_result": by_result,
            "by_component": by_component,
            "recent_failures": [
                {
                    "component": a.component,
                    "timestamp": a.timestamp.isoformat(),
                    "error": a.error,
                }
                for a in self.recovery_history[-5:]
                if a.result == RecoveryResult.FAILED
            ],
        }

    def reset_attempts(self, component: Optional[str] = None) -> None:
        """Reset attempt counts."""
        if component:
            self.attempt_counts.pop(component, None)
            self.last_recovery.pop(component, None)
        else:
            self.attempt_counts.clear()
            self.last_recovery.clear()


# ==================== Convenience Functions ====================


async def create_auto_recovery(
    notification_callback: Optional[Callable] = None,
) -> AutoRecovery:
    """Create pre-configured auto-recovery service."""

    config = RecoveryConfig(
        max_attempts=3,
        backoff_base=2.0,
        backoff_max=300.0,
        cooldown_period=60.0,
        escalation_threshold=5,
    )

    return AutoRecovery(
        config=config,
        notification_callback=notification_callback,
    )


async def recover_trading_bot(
    recovery: AutoRecovery,
    script: str = "scripts/trading/run_paper_trading.py",
) -> RecoveryResult:
    """Recover the trading bot process."""
    return await recovery.recover(
        component="trading_bot",
        component_type="process",
        context={
            "script": script,
            "log_file": "data/logs/trading_bot.log",
        },
    )


async def recover_data_feed(
    recovery: AutoRecovery,
    provider: str = "binance",
) -> RecoveryResult:
    """Recover a data feed."""
    return await recovery.recover(
        component=f"data_feed_{provider}",
        component_type="data_feed",
        context={"provider": provider},
    )
