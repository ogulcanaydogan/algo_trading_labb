"""
Interactive Telegram Bot for Trading Control

Phase 10: Telegram bot that allows users to control and monitor
the trading system via chat commands.

Commands:
/start - Welcome message and available commands
/status - System status overview
/positions - Current open positions
/pnl - Today's P&L summary
/trades - Recent trades
/regime - Current market regime
/risk - Risk metrics
/strategies - Strategy status
/enable <strategy> - Enable a strategy
/disable <strategy> - Disable a strategy
/pause - Pause trading
/resume - Resume trading
/kill - Emergency kill switch
/report - Generate performance report
/help - Show help message
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class BotCommandAccess(Enum):
    """Access levels for commands."""
    PUBLIC = "public"      # Anyone can use
    USER = "user"          # Registered users only
    ADMIN = "admin"        # Admins only


@dataclass
class BotCommand:
    """Definition of a bot command."""
    command: str
    description: str
    access: BotCommandAccess
    handler: Optional[Callable] = None


class TelegramTradingBot:
    """
    Interactive Telegram bot for trading system control.

    Features:
    - Command-based interaction
    - Access control (admin/user/public)
    - Real-time status queries
    - Trading controls (pause/resume/kill)
    - Strategy management
    - Performance reports
    """

    def __init__(
        self,
        bot_token: str,
        admin_chat_ids: List[str],
        allowed_chat_ids: Optional[List[str]] = None,
        orchestrator=None,
        notification_manager=None
    ):
        """
        Initialize the Telegram trading bot.

        Args:
            bot_token: Telegram bot token
            admin_chat_ids: Chat IDs with admin access
            allowed_chat_ids: Chat IDs allowed to use the bot (None = admins only)
            orchestrator: Trading orchestrator instance
            notification_manager: Notification manager instance
        """
        self.bot_token = bot_token
        self.admin_chat_ids = set(str(cid).strip() for cid in admin_chat_ids if cid)
        self.allowed_chat_ids = set(str(cid).strip() for cid in (allowed_chat_ids or [])) | self.admin_chat_ids

        self.orchestrator = orchestrator
        self.notification_manager = notification_manager

        # Command registry
        self.commands: Dict[str, BotCommand] = {}
        self._register_commands()

        # State
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._last_update_id = 0

        # Rate limiting per user
        self._user_last_command: Dict[str, datetime] = {}
        self._command_cooldown_seconds = 2

        logger.info(
            f"TelegramTradingBot initialized with {len(self.admin_chat_ids)} admins, "
            f"{len(self.allowed_chat_ids)} allowed users"
        )

    def _register_commands(self):
        """Register all bot commands."""
        commands = [
            BotCommand("/start", "Start and show welcome message", BotCommandAccess.PUBLIC),
            BotCommand("/help", "Show available commands", BotCommandAccess.PUBLIC),
            BotCommand("/status", "System status overview", BotCommandAccess.USER),
            BotCommand("/positions", "Current open positions", BotCommandAccess.USER),
            BotCommand("/pnl", "Today's P&L summary", BotCommandAccess.USER),
            BotCommand("/trades", "Recent trades", BotCommandAccess.USER),
            BotCommand("/regime", "Current market regime", BotCommandAccess.USER),
            BotCommand("/risk", "Risk metrics", BotCommandAccess.USER),
            BotCommand("/strategies", "Strategy status", BotCommandAccess.USER),
            BotCommand("/enable", "Enable a strategy", BotCommandAccess.ADMIN),
            BotCommand("/disable", "Disable a strategy", BotCommandAccess.ADMIN),
            BotCommand("/pause", "Pause trading", BotCommandAccess.ADMIN),
            BotCommand("/resume", "Resume trading", BotCommandAccess.ADMIN),
            BotCommand("/kill", "Emergency kill switch", BotCommandAccess.ADMIN),
            BotCommand("/report", "Generate performance report", BotCommandAccess.USER),
            BotCommand("/alerts", "Toggle alert settings", BotCommandAccess.USER),
            # AI Intelligence commands
            BotCommand("/explain", "Explain last trade decision", BotCommandAccess.USER),
            BotCommand("/ai", "AI brain status and insights", BotCommandAccess.USER),
            BotCommand("/learning", "Learning stats and patterns", BotCommandAccess.USER),
        ]

        for cmd in commands:
            self.commands[cmd.command] = cmd

    def inject_orchestrator(self, orchestrator):
        """Inject the trading orchestrator."""
        self.orchestrator = orchestrator

    def inject_notification_manager(self, notification_manager):
        """Inject the notification manager."""
        self.notification_manager = notification_manager

    async def start(self):
        """Start the bot polling loop."""
        if self._running:
            logger.warning("Bot already running")
            return

        if not self.bot_token:
            logger.warning("Bot token not configured, bot not started")
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_updates())
        logger.info("Telegram bot started")

        # Send startup message to admins
        await self._notify_admins("🤖 Trading bot started and ready for commands.")

    async def stop(self):
        """Stop the bot."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        await self._notify_admins("🛑 Trading bot stopped.")
        logger.info("Telegram bot stopped")

    async def _poll_updates(self):
        """Poll for updates from Telegram."""
        import aiohttp

        while self._running:
            try:
                updates = await self._get_updates()

                for update in updates:
                    await self._process_update(update)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll error: {e}")
                await asyncio.sleep(5)

    async def _get_updates(self) -> List[Dict]:
        """Get updates from Telegram API."""
        import aiohttp

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {
            "offset": self._last_update_id + 1,
            "timeout": 30,
            "allowed_updates": ["message"]
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                if not data.get("ok"):
                    return []

                updates = data.get("result", [])

                if updates:
                    self._last_update_id = max(u["update_id"] for u in updates)

                return updates

    async def _process_update(self, update: Dict):
        """Process a single update."""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "")
        user = message.get("from", {})
        username = user.get("username", "unknown")

        if not text.startswith("/"):
            return

        # Parse command and arguments
        parts = text.split()
        command = parts[0].lower().split("@")[0]  # Remove @bot_name suffix
        args = parts[1:] if len(parts) > 1 else []

        logger.info(f"Command from {username} ({chat_id}): {command} {args}")

        # Check access
        access_level = self._get_access_level(chat_id)
        if access_level is None:
            await self._send_message(chat_id, "⛔ You are not authorized to use this bot.")
            return

        # Rate limiting
        if not self._check_rate_limit(chat_id):
            return

        # Handle command
        await self._handle_command(chat_id, command, args, access_level, username)

    def _get_access_level(self, chat_id: str) -> Optional[BotCommandAccess]:
        """Get access level for a chat ID."""
        if chat_id in self.admin_chat_ids:
            return BotCommandAccess.ADMIN
        elif chat_id in self.allowed_chat_ids:
            return BotCommandAccess.USER
        else:
            return None

    def _check_rate_limit(self, chat_id: str) -> bool:
        """Check if user is rate limited."""
        now = datetime.now()
        last = self._user_last_command.get(chat_id)

        if last and (now - last).total_seconds() < self._command_cooldown_seconds:
            return False

        self._user_last_command[chat_id] = now
        return True

    async def _handle_command(
        self,
        chat_id: str,
        command: str,
        args: List[str],
        access_level: BotCommandAccess,
        username: str
    ):
        """Handle a command."""
        cmd_def = self.commands.get(command)

        if not cmd_def:
            await self._send_message(chat_id, f"Unknown command: {command}\nUse /help to see available commands.")
            return

        # Check access level
        if cmd_def.access == BotCommandAccess.ADMIN and access_level != BotCommandAccess.ADMIN:
            await self._send_message(chat_id, "⛔ This command requires admin access.")
            return

        # Dispatch to handler
        handler_name = f"_cmd{command.replace('/', '_')}"
        handler = getattr(self, handler_name, None)

        if handler:
            try:
                await handler(chat_id, args, username)
            except Exception as e:
                logger.error(f"Command handler error: {e}")
                await self._send_message(chat_id, f"❌ Error: {e}")
        else:
            await self._send_message(chat_id, "Command not implemented yet.")

    async def _send_message(self, chat_id: str, text: str, parse_mode: str = "HTML"):
        """Send a message to a chat."""
        import aiohttp

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Failed to send message: {error}")

    async def _notify_admins(self, message: str):
        """Send a message to all admins."""
        for chat_id in self.admin_chat_ids:
            try:
                await self._send_message(chat_id, message)
            except Exception as e:
                logger.error(f"Failed to notify admin {chat_id}: {e}")

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def _cmd_start(self, chat_id: str, args: List[str], username: str):
        """Handle /start command."""
        message = f"""
🤖 <b>Trading Bot Control</b>

Welcome, {username}!

This bot allows you to monitor and control your algorithmic trading system.

<b>Quick Commands:</b>
/status - System status
/positions - Open positions
/pnl - Today's P&L
/help - All commands

<i>Use /help for full command list.</i>
        """
        await self._send_message(chat_id, message.strip())

    async def _cmd_help(self, chat_id: str, args: List[str], username: str):
        """Handle /help command."""
        access = self._get_access_level(chat_id)

        lines = ["<b>Available Commands:</b>\n"]

        # Group by access level
        user_cmds = []
        admin_cmds = []

        for cmd, cmd_def in sorted(self.commands.items()):
            if cmd_def.access == BotCommandAccess.ADMIN:
                admin_cmds.append(f"{cmd} - {cmd_def.description}")
            else:
                user_cmds.append(f"{cmd} - {cmd_def.description}")

        lines.append("<b>📊 Monitoring:</b>")
        lines.extend(user_cmds)

        if access == BotCommandAccess.ADMIN:
            lines.append("\n<b>🔐 Admin Controls:</b>")
            lines.extend(admin_cmds)

        await self._send_message(chat_id, "\n".join(lines))

    async def _cmd_status(self, chat_id: str, args: List[str], username: str):
        """Handle /status command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        status = self.orchestrator.get_status()
        health = self.orchestrator.get_health()

        state_emoji = {
            "running": "🟢",
            "paused": "🟡",
            "stopped": "🔴",
            "error": "❌"
        }

        emoji = state_emoji.get(status["state"], "⚪")

        message = f"""
{emoji} <b>System Status</b>

<b>State:</b> {status['state'].upper()}
<b>Mode:</b> {status['mode']}
<b>Uptime:</b> {health.uptime_seconds / 3600:.1f} hours
<b>Regime:</b> {status['current_regime']}

<b>Today's Activity:</b>
• Decisions: {health.decisions_today}
• Trades: {health.trades_today}
• Errors: {health.errors_today}

<b>Active Strategies:</b> {len(status['active_strategies'])}
{', '.join(status['active_strategies'][:5]) or 'None'}

<b>Components:</b>
• Risk Guardian: {'✅' if status['components']['risk_guardian'] else '❌'}
• Execution: {'✅' if status['components']['execution_engine'] else '❌'}
• AI Integration: {'✅' if status['components']['ai_integration'] else '❌'}
        """
        await self._send_message(chat_id, message.strip())

    async def _cmd_positions(self, chat_id: str, args: List[str], username: str):
        """Handle /positions command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        # Get positions from orchestrator (placeholder - would connect to actual data)
        positions = []  # Would come from execution engine

        if not positions:
            await self._send_message(chat_id, "📊 No open positions.")
            return

        lines = ["<b>📊 Open Positions</b>\n"]

        for pos in positions:
            pnl_emoji = "🟢" if pos.get("pnl", 0) >= 0 else "🔴"
            lines.append(
                f"{pnl_emoji} <b>{pos['symbol']}</b>: {pos['quantity']} @ ${pos['entry_price']:.2f}\n"
                f"   P&L: ${pos.get('pnl', 0):+.2f} ({pos.get('pnl_percent', 0):+.2f}%)"
            )

        await self._send_message(chat_id, "\n".join(lines))

    async def _cmd_pnl(self, chat_id: str, args: List[str], username: str):
        """Handle /pnl command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        # Get P&L data (placeholder)
        health = self.orchestrator.get_health()

        pnl = health.daily_pnl
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        message = f"""
{pnl_emoji} <b>Today's P&L</b>

<b>Total P&L:</b> ${pnl:+,.2f}
<b>Trades:</b> {health.trades_today}
<b>Open Positions:</b> {health.open_positions}
<b>Current Drawdown:</b> {health.current_drawdown:.2%}
        """
        await self._send_message(chat_id, message.strip())

    async def _cmd_trades(self, chat_id: str, args: List[str], username: str):
        """Handle /trades command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        # Get recent trades
        decisions = [d for d in self.orchestrator.recent_decisions if d.executed][-10:]

        if not decisions:
            await self._send_message(chat_id, "📊 No recent trades.")
            return

        lines = ["<b>📊 Recent Trades</b>\n"]

        for d in reversed(decisions):
            side_emoji = "🟢" if d.action == "buy" else "🔴"
            lines.append(
                f"{side_emoji} {d.action.upper()} {d.quantity} {d.symbol} @ ${d.execution_price or d.price:.2f}\n"
                f"   <i>{d.timestamp.strftime('%H:%M:%S')} via {d.strategy_name}</i>"
            )

        await self._send_message(chat_id, "\n".join(lines))

    async def _cmd_regime(self, chat_id: str, args: List[str], username: str):
        """Handle /regime command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        regime = self.orchestrator.current_regime

        regime_emoji = {
            "trending_bullish": "📈",
            "trending_bearish": "📉",
            "mean_reverting": "↔️",
            "high_volatility": "⚡",
            "low_volatility": "😴",
            "crisis": "🚨"
        }

        emoji = regime_emoji.get(regime, "❓")

        message = f"""
{emoji} <b>Market Regime</b>

<b>Current:</b> {regime.replace('_', ' ').title()}

<b>Strategy Recommendations:</b>
{self._get_regime_recommendations(regime)}
        """
        await self._send_message(chat_id, message.strip())

    def _get_regime_recommendations(self, regime: str) -> str:
        """Get strategy recommendations for a regime."""
        recommendations = {
            "trending_bullish": "• Momentum strategies favored\n• Increase position sizes\n• Trail stops loosely",
            "trending_bearish": "• Short strategies or cash\n• Reduce exposure\n• Tighten stops",
            "mean_reverting": "• Mean reversion strategies\n• Range trading\n• Standard position sizes",
            "high_volatility": "• Reduce position sizes\n• Wider stops\n• Consider options hedging",
            "low_volatility": "• Breakout strategies\n• Normal position sizes\n• Watch for regime change",
            "crisis": "• Minimum exposure\n• Cash is king\n• Wait for stability"
        }
        return recommendations.get(regime, "• Monitor closely\n• Use caution")

    async def _cmd_risk(self, chat_id: str, args: List[str], username: str):
        """Handle /risk command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        health = self.orchestrator.get_health()

        # Risk level indicators
        dd_level = "🟢" if health.current_drawdown < 0.03 else "🟡" if health.current_drawdown < 0.07 else "🔴"

        message = f"""
🛡️ <b>Risk Metrics</b>

<b>Drawdown:</b> {dd_level} {health.current_drawdown:.2%}
<b>Open Positions:</b> {health.open_positions}
<b>Daily P&L:</b> ${health.daily_pnl:+,.2f}

<b>Risk Guardian:</b> {'✅ Active' if health.risk_guardian_active else '❌ Inactive'}

<b>Warnings:</b>
{chr(10).join(['• ' + w for w in health.warnings]) or '• None'}
        """
        await self._send_message(chat_id, message.strip())

    async def _cmd_strategies(self, chat_id: str, args: List[str], username: str):
        """Handle /strategies command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        strategies = self.orchestrator.active_strategies

        lines = ["<b>📋 Strategy Status</b>\n"]

        for name, enabled in sorted(strategies.items()):
            status = "✅ Enabled" if enabled else "⏸️ Disabled"
            lines.append(f"• <b>{name}</b>: {status}")

        lines.append(f"\nTotal: {len(strategies)} strategies")
        lines.append(f"Active: {sum(1 for e in strategies.values() if e)}")

        await self._send_message(chat_id, "\n".join(lines))

    async def _cmd_enable(self, chat_id: str, args: List[str], username: str):
        """Handle /enable command."""
        if not args:
            await self._send_message(chat_id, "Usage: /enable <strategy_name>")
            return

        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        strategy_name = args[0]
        success = self.orchestrator.enable_strategy(strategy_name)

        if success:
            await self._send_message(chat_id, f"✅ Strategy <b>{strategy_name}</b> enabled.")
            await self._notify_admins(f"ℹ️ {username} enabled strategy: {strategy_name}")
        else:
            await self._send_message(chat_id, f"❌ Failed to enable strategy: {strategy_name}")

    async def _cmd_disable(self, chat_id: str, args: List[str], username: str):
        """Handle /disable command."""
        if not args:
            await self._send_message(chat_id, "Usage: /disable <strategy_name>")
            return

        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        strategy_name = args[0]
        success = self.orchestrator.disable_strategy(strategy_name)

        if success:
            await self._send_message(chat_id, f"⏸️ Strategy <b>{strategy_name}</b> disabled.")
            await self._notify_admins(f"ℹ️ {username} disabled strategy: {strategy_name}")
        else:
            await self._send_message(chat_id, f"❌ Failed to disable strategy: {strategy_name}")

    async def _cmd_pause(self, chat_id: str, args: List[str], username: str):
        """Handle /pause command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        reason = " ".join(args) if args else f"Paused by {username} via Telegram"
        await self.orchestrator.pause(reason=reason)

        await self._send_message(chat_id, f"⏸️ Trading paused: {reason}")
        await self._notify_admins(f"⏸️ Trading paused by {username}: {reason}")

    async def _cmd_resume(self, chat_id: str, args: List[str], username: str):
        """Handle /resume command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        await self.orchestrator.resume()

        await self._send_message(chat_id, "▶️ Trading resumed.")
        await self._notify_admins(f"▶️ Trading resumed by {username}")

    async def _cmd_kill(self, chat_id: str, args: List[str], username: str):
        """Handle /kill command (emergency stop)."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        # Require confirmation
        if not args or args[0].lower() != "confirm":
            await self._send_message(
                chat_id,
                "⚠️ <b>WARNING: Emergency Kill Switch</b>\n\n"
                "This will immediately stop all trading.\n\n"
                "To confirm, type: /kill confirm"
            )
            return

        reason = f"Kill switch activated by {username} via Telegram"
        await self.orchestrator.kill_switch(action="stop", reason=reason)

        await self._send_message(chat_id, "🛑 <b>KILL SWITCH ACTIVATED</b>\n\nAll trading has been stopped.")
        await self._notify_admins(f"🚨 KILL SWITCH activated by {username}")

        if self.notification_manager:
            await self.notification_manager.notify_kill_switch("stop", reason)

    async def _cmd_report(self, chat_id: str, args: List[str], username: str):
        """Handle /report command."""
        if not self.orchestrator:
            await self._send_message(chat_id, "⚠️ Orchestrator not connected.")
            return

        # Generate mini report
        health = self.orchestrator.get_health()
        status = self.orchestrator.get_status()

        message = f"""
📋 <b>Performance Report</b>

<b>System:</b> {status['state'].upper()} | {status['mode']}
<b>Uptime:</b> {health.uptime_seconds / 3600:.1f} hours

<b>Today's Activity:</b>
• Decisions Made: {health.decisions_today}
• Trades Executed: {health.trades_today}
• Win Rate: N/A
• Errors: {health.errors_today}

<b>Risk Status:</b>
• Current Drawdown: {health.current_drawdown:.2%}
• Daily P&L: ${health.daily_pnl:+,.2f}
• Open Positions: {health.open_positions}

<b>Active Strategies:</b> {len(status['active_strategies'])}

<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        await self._send_message(chat_id, message.strip())

    async def _cmd_alerts(self, chat_id: str, args: List[str], username: str):
        """Handle /alerts command."""
        message = """
🔔 <b>Alert Settings</b>

Current notification preferences can be configured via environment variables:

• TELEGRAM_ENABLED=true
• TELEGRAM_CHAT_IDS=your_chat_id

Alert categories:
• 💰 Trades - Trade executions
• 🛡️ Risk - Risk alerts
• 🔄 Regime - Market regime changes
• ❌ Errors - System errors
• 📋 Reports - Daily reports

Contact admin to modify settings.
        """
        await self._send_message(chat_id, message.strip())

    # =========================================================================
    # AI Intelligence Command Handlers
    # =========================================================================

    async def _cmd_explain(self, chat_id: str, args: List[str], username: str):
        """Handle /explain command - Explain last trade decision."""
        try:
            from bot.intelligence import get_intelligent_brain

            brain = get_intelligent_brain()
            summary = brain.get_summary()

            # Get last explanation from trade explainer
            explainer_stats = summary.get("trade_explainer", {})
            llm_stats = explainer_stats.get("llm_stats", {})

            # Check pattern memory for recent trades
            pattern_stats = summary.get("pattern_memory", {})

            if pattern_stats.get("total_patterns", 0) == 0:
                message = """
🧠 <b>Trade Explanation</b>

No trades have been made yet. The AI brain will explain each trade when they happen.

<b>What it explains:</b>
• Why the trade was made
• Market regime at entry
• Confidence level
• Risk assessment
• News sentiment (if available)

Explanations are sent automatically via Telegram when trades execute.
                """
            else:
                recent_perf = summary.get("learner", {}).get("recent_performance", {})
                win_rate = pattern_stats.get("overall_win_rate", 0)

                message = f"""
🧠 <b>Trade Explanation System</b>

<b>Trades Analyzed:</b> {pattern_stats.get('total_patterns', 0)}
<b>Win Rate:</b> {win_rate:.1%}
<b>Avg P&L:</b> {pattern_stats.get('average_pnl_pct', 0):.2f}%

<b>LLM Usage:</b>
• Claude: {llm_stats.get('claude_requests', 0)} requests
• Ollama: {llm_stats.get('ollama_requests', 0)} requests
• Cost Today: ${llm_stats.get('claude_cost_today', 0):.4f}

<i>Trade explanations are sent automatically when trades execute.</i>
                """

            await self._send_message(chat_id, message.strip())

        except ImportError:
            await self._send_message(chat_id, "⚠️ AI Intelligence module not available.")
        except Exception as e:
            await self._send_message(chat_id, f"❌ Error: {e}")

    async def _cmd_ai(self, chat_id: str, args: List[str], username: str):
        """Handle /ai command - AI brain status and insights."""
        try:
            from bot.intelligence import get_intelligent_brain

            brain = get_intelligent_brain()
            health = brain.health_check()

            # Format LLM status
            llm = health.get("llm_router", {})
            claude_status = "✅" if llm.get("claude_available") else "❌"
            ollama_status = "✅" if llm.get("ollama_available") else "❌"

            # Format regime
            regime_info = health.get("regime_adapter", {})
            regime = regime_info.get("current_regime", "unknown")
            confidence = regime_info.get("confidence", 0)

            # Format news
            news_info = health.get("news_reasoner", {})
            sentiment = news_info.get("current_sentiment", 0)
            sentiment_emoji = "📈" if sentiment > 0.1 else "📉" if sentiment < -0.1 else "➡️"

            # Budget info
            budget = llm.get("claude_budget_remaining", 0)

            message = f"""
🤖 <b>AI Trading Brain Status</b>

<b>LLM Backends:</b>
• Claude API: {claude_status} (${budget:.2f} remaining)
• Ollama (local): {ollama_status}
• Rule-based: ✅

<b>Market Regime:</b>
• Current: {regime.replace('_', ' ').title()}
• Confidence: {confidence:.1%}

<b>News Sentiment:</b> {sentiment_emoji} {sentiment:+.2f}

<b>Learning Status:</b>
• Patterns Stored: {health.get('pattern_memory', {}).get('total_patterns', 0)}
• Win Rate: {health.get('pattern_memory', {}).get('overall_win_rate', 0):.1%}

<b>Telegram:</b> {'✅ Connected' if health.get('trade_explainer', {}).get('telegram_connected') else '❌ Disconnected'}
            """
            await self._send_message(chat_id, message.strip())

        except ImportError:
            await self._send_message(chat_id, "⚠️ AI Intelligence module not available.")
        except Exception as e:
            await self._send_message(chat_id, f"❌ Error: {e}")

    async def _cmd_learning(self, chat_id: str, args: List[str], username: str):
        """Handle /learning command - Learning stats and patterns."""
        try:
            from bot.intelligence import get_intelligent_brain, PatternMemory

            brain = get_intelligent_brain()
            summary = brain.get_summary()

            pattern_stats = summary.get("pattern_memory", {})
            learner_stats = summary.get("learner", {})

            # Get pattern breakdown by regime if available
            memory = PatternMemory()
            regime_summary = memory.get_summary()

            total = pattern_stats.get("total_patterns", 0)
            profitable = pattern_stats.get("profitable_patterns", 0)
            win_rate = pattern_stats.get("overall_win_rate", 0)
            avg_pnl = pattern_stats.get("average_pnl_pct", 0)

            message = f"""
📚 <b>Learning Statistics</b>

<b>Pattern Memory:</b>
• Total Patterns: {total}
• Profitable: {profitable}
• Win Rate: {win_rate:.1%}
• Avg P&L: {avg_pnl:+.2f}%

<b>Confidence Adjustments:</b>
• Active: {learner_stats.get('confidence_adjustments', 0)}
• Learning Rate: {learner_stats.get('learning_rate', 0.1)}

<b>Coverage:</b>
• Symbols: {pattern_stats.get('unique_symbols', 0)}
• Regimes: {pattern_stats.get('unique_regimes', 0)}

<b>Recent Performance:</b>
{learner_stats.get('recent_performance', {}).get('message', 'No recent trades')}

<i>The brain learns from every trade and adjusts confidence thresholds based on pattern success rates.</i>
            """
            await self._send_message(chat_id, message.strip())

        except ImportError:
            await self._send_message(chat_id, "⚠️ AI Intelligence module not available.")
        except Exception as e:
            await self._send_message(chat_id, f"❌ Error: {e}")


def create_telegram_bot(
    orchestrator=None,
    notification_manager=None
) -> Optional[TelegramTradingBot]:
    """
    Create a Telegram trading bot from environment variables.

    Returns:
        TelegramTradingBot if configured, None otherwise
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    admin_ids = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").split(",")
    allowed_ids = os.getenv("TELEGRAM_CHAT_IDS", "").split(",")

    if not bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not set, Telegram bot disabled")
        return None

    return TelegramTradingBot(
        bot_token=bot_token,
        admin_chat_ids=admin_ids,
        allowed_chat_ids=allowed_ids,
        orchestrator=orchestrator,
        notification_manager=notification_manager
    )


if __name__ == "__main__":
    # Demo - would need actual bot token to run
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def demo():
        print("=== Telegram Trading Bot Demo ===")
        print("To use, set environment variables:")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_ADMIN_CHAT_ID=your_chat_id")
        print("  TELEGRAM_CHAT_IDS=allowed_chat_ids")

        # Create bot (won't start without token)
        bot = create_telegram_bot()

        if bot:
            print(f"\nBot created with {len(bot.commands)} commands:")
            for cmd, cmd_def in bot.commands.items():
                print(f"  {cmd}: {cmd_def.description} [{cmd_def.access.value}]")
        else:
            print("\nBot not configured (missing token)")

    asyncio.run(demo())
