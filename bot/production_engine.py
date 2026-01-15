"""
Production Trading Engine

Wraps the unified trading engine with production safety features:
- Health monitoring
- Auto-recovery
- Circuit breaker
- Telegram notifications
"""

import asyncio
import logging
import os
import signal
from datetime import datetime
from typing import Any, Callable, Optional

from bot.auto_recovery import AutoRecovery, RecoveryConfig, RecoveryResult
from bot.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitConfig,
    CircuitOpenError,
    CircuitState,
)
from bot.health_monitor import (
    ComponentHealth,
    ComponentType,
    HealthMonitor,
    HealthStatus,
    SystemHealth,
)
from bot.unified_engine import EngineConfig, UnifiedTradingEngine

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram notification service."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        self._session = None

        if not self.enabled:
            logger.warning("Telegram notifications disabled (missing credentials)")

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram."""
        if not self.enabled:
            logger.info(f"[Telegram disabled] {message}")
            return False

        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        logger.debug(f"Telegram sent: {message[:50]}...")
                        return True
                    else:
                        logger.error(f"Telegram error: {resp.status}")
                        return False

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def send_alert(self, title: str, details: str, level: str = "warning") -> bool:
        """Send formatted alert."""
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
            "critical": "🔴",
            "success": "✅",
        }.get(level, "📢")

        message = f"{emoji} <b>{title}</b>\n\n{details}\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        return await self.send(message)


class ProductionEngine:
    """
    Production-hardened trading engine.

    Wraps UnifiedTradingEngine with:
    - Health monitoring with automatic checks
    - Circuit breaker to stop trading on errors
    - Auto-recovery for failed components
    - Telegram notifications for alerts
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ):
        self.config = config or EngineConfig()

        # Core engine
        self.engine: Optional[UnifiedTradingEngine] = None

        # Safety components
        self.health_monitor: Optional[HealthMonitor] = None
        self.circuit_breaker_manager: Optional[CircuitBreakerManager] = None
        self.auto_recovery: Optional[AutoRecovery] = None

        # Circuit breakers
        self.trading_circuit: Optional[CircuitBreaker] = None
        self.data_feed_circuit: Optional[CircuitBreaker] = None
        self.api_circuit: Optional[CircuitBreaker] = None

        # Notifications
        self.notifier = TelegramNotifier(telegram_token, telegram_chat_id)

        # State
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[SystemHealth] = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize core engine
            self.engine = UnifiedTradingEngine(self.config)
            initialized = await self.engine.initialize()

            if not initialized:
                logger.error("Failed to initialize trading engine")
                return False

            # Initialize circuit breakers
            self.circuit_breaker_manager = CircuitBreakerManager(
                on_any_trip=self._on_circuit_trip
            )

            self.trading_circuit = self.circuit_breaker_manager.create(
                "trading",
                CircuitConfig(
                    error_rate_threshold=0.3,
                    consecutive_failures_threshold=3,
                    latency_threshold_ms=10000,
                    reset_timeout_seconds=60,
                ),
            )

            self.data_feed_circuit = self.circuit_breaker_manager.create(
                "data_feed",
                CircuitConfig(
                    error_rate_threshold=0.5,
                    consecutive_failures_threshold=5,
                    latency_threshold_ms=5000,
                    reset_timeout_seconds=30,
                ),
            )

            self.api_circuit = self.circuit_breaker_manager.create(
                "api",
                CircuitConfig(
                    error_rate_threshold=0.4,
                    consecutive_failures_threshold=4,
                    latency_threshold_ms=3000,
                    reset_timeout_seconds=45,
                ),
            )

            # Initialize auto-recovery
            self.auto_recovery = AutoRecovery(
                config=RecoveryConfig(
                    max_attempts=3,
                    backoff_base=2.0,
                    backoff_max=300.0,
                    cooldown_period=60.0,
                    escalation_threshold=5,
                ),
                notification_callback=self._on_recovery_event,
                escalation_callback=self._on_escalation,
            )

            # Initialize health monitor
            self.health_monitor = HealthMonitor(
                check_interval=30.0,
                alert_threshold=3,
                recovery_callback=self._on_health_recovery_needed,
                notification_callback=self._on_health_notification,
            )

            # Register health checks
            await self._register_health_checks()

            logger.info("Production engine initialized")
            await self.notifier.send_alert(
                "Trading Bot Started",
                f"Mode: {self.config.initial_mode.value}\n"
                f"Capital: ${self.config.initial_capital:,.2f}\n"
                f"Symbols: {', '.join(self.config.symbols)}",
                "success",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize production engine: {e}")
            return False

    async def _register_health_checks(self) -> None:
        """Register all health checks."""
        from bot.health_monitor import (
            create_api_health_check,
            create_file_freshness_check,
            create_memory_health_check,
            create_disk_health_check,
        )

        # API health
        self.health_monitor.register_component(
            "trading_api",
            ComponentType.API,
            create_api_health_check("http://localhost:8000/health"),
        )

        # State file freshness
        state_file = str(self.config.data_dir / "state.json")
        self.health_monitor.register_component(
            "state_file",
            ComponentType.DATA_FEED,
            create_file_freshness_check(state_file, max_age_seconds=600),
        )

        # System resources
        self.health_monitor.register_component(
            "memory",
            ComponentType.OTHER,
            create_memory_health_check(max_usage_percent=85.0),
        )

        self.health_monitor.register_component(
            "disk",
            ComponentType.OTHER,
            create_disk_health_check("/", min_free_gb=5.0),
        )

    async def start(self) -> None:
        """Start the production engine."""
        if self._running:
            logger.warning("Production engine already running")
            return

        if not self.engine:
            raise RuntimeError("Engine not initialized")

        self._running = True

        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitoring_loop())

        # Start trading engine
        await self.engine.start()

        logger.info("Production engine started")

    async def stop(self) -> None:
        """Stop the production engine gracefully."""
        if not self._running:
            return

        logger.info("Stopping production engine...")
        self._running = False

        # Stop health monitoring
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop()

        # Stop trading engine
        if self.engine:
            await self.engine.stop()

        await self.notifier.send_alert(
            "Trading Bot Stopped",
            "Graceful shutdown completed",
            "info",
        )

        logger.info("Production engine stopped")

    async def emergency_stop(self, reason: str) -> None:
        """Emergency stop all trading."""
        logger.critical(f"EMERGENCY STOP: {reason}")

        # Open all circuits
        if self.circuit_breaker_manager:
            await self.circuit_breaker_manager.force_open_all(reason)

        # Stop engine
        if self.engine:
            await self.engine.emergency_stop(reason)

        self._running = False

        await self.notifier.send_alert(
            "EMERGENCY STOP",
            f"Reason: {reason}\n\nAll trading halted immediately.",
            "critical",
        )

    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while self._running:
            try:
                if self.health_monitor:
                    health = await self.health_monitor.check_all()
                    self._last_health_check = health

                    # Check overall health
                    if health.status == HealthStatus.CRITICAL:
                        self._consecutive_failures += 1

                        if self._consecutive_failures >= self._max_consecutive_failures:
                            alerts = ", ".join(health.alerts) if health.alerts else "Unknown"
                            await self.emergency_stop(
                                f"Too many consecutive health failures: {alerts}"
                            )
                    else:
                        self._consecutive_failures = 0

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _on_circuit_trip(
        self,
        circuit_name: str,
        reason: str,
        details: str,
    ) -> None:
        """Handle circuit breaker trip."""
        logger.warning(f"Circuit {circuit_name} tripped: {reason} - {details}")

        await self.notifier.send_alert(
            f"Circuit Breaker: {circuit_name}",
            f"<b>Reason:</b> {reason}\n<b>Details:</b> {details}\n\n"
            "Trading through this circuit is temporarily blocked.",
            "warning",
        )

        # If trading circuit trips, pause engine
        if circuit_name == "trading" and self.engine:
            logger.warning("Trading circuit tripped - pausing engine")
            # Don't stop completely, just let the circuit block trades

    async def _on_recovery_event(self, message: str, context: dict) -> None:
        """Handle recovery event."""
        logger.info(f"Recovery event: {message}")
        await self.notifier.send_alert("Recovery Event", message, "info")

    async def _on_escalation(self, component: str, reason: str) -> None:
        """Handle escalation (unrecoverable failure)."""
        logger.critical(f"ESCALATION: {component} - {reason}")

        await self.notifier.send_alert(
            "ESCALATION - Manual Intervention Required",
            f"<b>Component:</b> {component}\n<b>Reason:</b> {reason}\n\n"
            "Automatic recovery has failed. Manual intervention required.",
            "critical",
        )

    async def _on_health_recovery_needed(
        self,
        component: str,
        health: ComponentHealth,
    ) -> None:
        """Handle health check failure - trigger recovery."""
        if not self.auto_recovery:
            return

        logger.warning(f"Health recovery needed for {component}: {health.status}")

        # Map component types to recovery types
        recovery_type_map = {
            ComponentType.EXCHANGE: "connection",
            ComponentType.API: "api",
            ComponentType.DATA_FEED: "data_feed",
            ComponentType.ORCHESTRATOR: "process",
        }

        recovery_type = recovery_type_map.get(health.component_type, "process")

        result = await self.auto_recovery.recover(
            component=component,
            component_type=recovery_type,
            context={"health": health},
        )

        if result == RecoveryResult.FAILED:
            logger.error(f"Recovery failed for {component}")

    async def _on_health_notification(self, message: str, context: dict) -> None:
        """Handle health notification."""
        level = context.get("level", "info")
        await self.notifier.send_alert("Health Alert", message, level)

    async def execute_with_circuit_breaker(
        self,
        circuit_name: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function through the appropriate circuit breaker."""
        circuit = self.circuit_breaker_manager.get(circuit_name) if self.circuit_breaker_manager else None

        if not circuit:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

        try:
            return await circuit.call(func, *args, **kwargs)
        except CircuitOpenError as e:
            logger.warning(f"Circuit {circuit_name} is open: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive production status."""
        engine_status = self.engine.get_status() if self.engine else {}

        circuit_status = (
            self.circuit_breaker_manager.get_all_status()
            if self.circuit_breaker_manager
            else {}
        )

        recovery_stats = (
            self.auto_recovery.get_statistics()
            if self.auto_recovery
            else {}
        )

        health_status = None
        if self._last_health_check:
            components = self._last_health_check.components
            health_status = {
                "overall": self._last_health_check.status.value,
                "healthy_count": sum(1 for c in components.values() if c.status == HealthStatus.HEALTHY),
                "degraded_count": sum(1 for c in components.values() if c.status == HealthStatus.DEGRADED),
                "unhealthy_count": sum(1 for c in components.values() if c.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]),
                "alerts": self._last_health_check.alerts,
            }

        return {
            "production": {
                "running": self._running,
                "consecutive_failures": self._consecutive_failures,
                "telegram_enabled": self.notifier.enabled,
            },
            "engine": engine_status,
            "circuits": circuit_status,
            "recovery": recovery_stats,
            "health": health_status,
        }


async def create_production_engine(
    config: Optional[EngineConfig] = None,
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
) -> ProductionEngine:
    """Create and initialize a production engine."""
    engine = ProductionEngine(
        config=config,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
    )
    await engine.initialize()
    return engine


def setup_signal_handlers(engine: ProductionEngine) -> None:
    """Setup OS signal handlers for graceful shutdown."""

    def handle_sigterm(signum, frame):
        logger.info("Received SIGTERM, initiating graceful shutdown...")
        asyncio.create_task(engine.stop())

    def handle_sigint(signum, frame):
        logger.info("Received SIGINT, initiating graceful shutdown...")
        asyncio.create_task(engine.stop())

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)


async def run_production_bot(
    config: Optional[EngineConfig] = None,
) -> None:
    """Main entry point for production trading bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting production trading bot...")

    engine = await create_production_engine(config)
    setup_signal_handlers(engine)

    try:
        await engine.start()

        # Keep running until stopped
        while engine._running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(run_production_bot())
