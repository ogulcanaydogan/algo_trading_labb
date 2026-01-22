"""
Telegram Bot Commands Module.

Provides interactive commands via Telegram:
- /status - Get bot status
- /balance - Check balance
- /positions - List open positions
- /trades - Recent trades
- /pnl - Today's P&L
- /stop - Pause trading
- /start - Resume trading
- /close [symbol] - Close a position
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path("data/unified_trading")


class TelegramCommandBot:
    """
    Telegram bot with trading commands.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._running = False
        self._last_update_id = 0

        if not self.token:
            logger.warning("Telegram bot token not configured")

    def _get_state(self) -> dict:
        """Load current state."""
        state_file = DATA_DIR / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {}

    def _get_trades(self, limit: int = 10) -> list:
        """Load recent trades."""
        trades_file = DATA_DIR / "trades.json"
        if trades_file.exists():
            with open(trades_file) as f:
                trades = json.load(f)
                return trades[-limit:]
        return []

    def _get_control(self) -> dict:
        """Load control state."""
        control_file = DATA_DIR / "control.json"
        if control_file.exists():
            with open(control_file) as f:
                return json.load(f)
        return {"paused": False}

    def _save_control(self, control: dict):
        """Save control state."""
        control_file = DATA_DIR / "control.json"
        control_file.parent.mkdir(parents=True, exist_ok=True)
        with open(control_file, "w") as f:
            json.dump(control, f, indent=2)

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram."""
        if not self.token or not self.chat_id:
            return False

        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                }) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def handle_status(self) -> str:
        """Handle /status command."""
        state = self._get_state()
        control = self._get_control()

        status = "🟢 RUNNING" if not control.get("paused") else "🔴 PAUSED"
        mode = state.get("mode", "unknown").upper()
        balance = state.get("balance", 0)
        initial = state.get("initial_capital", 10000)
        pnl = balance - initial
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0
        positions = len(state.get("positions", {}))

        return f"""
<b>📊 Bot Status</b>

Status: {status}
Mode: {mode}
Balance: <b>${balance:,.2f}</b>
Total P&L: <b>${pnl:+,.2f}</b> ({pnl_pct:+.2f}%)
Open Positions: {positions}
Updated: {datetime.now().strftime('%H:%M:%S')}
"""

    def handle_balance(self) -> str:
        """Handle /balance command."""
        state = self._get_state()

        balance = state.get("balance", 0)
        initial = state.get("initial_capital", 10000)
        pnl = balance - initial
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0

        # Calculate position values
        positions = state.get("positions", {})
        pos_value = sum(
            p.get("quantity", 0) * p.get("entry_price", 0)
            for p in positions.values()
        )

        emoji = "📈" if pnl >= 0 else "📉"

        return f"""
<b>💰 Balance</b>

Cash: ${balance:,.2f}
Positions: ${pos_value:,.2f}
Total: ${balance + pos_value:,.2f}

{emoji} P&L: <b>${pnl:+,.2f}</b> ({pnl_pct:+.2f}%)
"""

    def handle_positions(self) -> str:
        """Handle /positions command."""
        state = self._get_state()
        positions = state.get("positions", {})

        if not positions:
            return "📭 <b>No open positions</b>"

        text = "<b>📊 Open Positions</b>\n\n"

        for symbol, pos in positions.items():
            side = pos.get("side", "long").upper()
            entry = pos.get("entry_price", 0)
            qty = pos.get("quantity", 0)
            upnl = pos.get("unrealized_pnl", 0)

            emoji = "🟢" if side == "LONG" else "🔴"
            pnl_emoji = "✅" if upnl >= 0 else "❌"

            text += f"{emoji} <b>{symbol}</b>\n"
            text += f"   Side: {side}\n"
            text += f"   Entry: ${entry:,.2f}\n"
            text += f"   Qty: {qty}\n"
            text += f"   {pnl_emoji} P&L: ${upnl:+,.2f}\n\n"

        return text

    def handle_trades(self, limit: int = 5) -> str:
        """Handle /trades command."""
        trades = self._get_trades(limit)

        if not trades:
            return "📭 <b>No recent trades</b>"

        text = f"<b>📜 Last {len(trades)} Trades</b>\n\n"

        for trade in reversed(trades):
            symbol = trade.get("symbol", "???")
            side = trade.get("side", "???").upper()
            pnl = trade.get("pnl", 0)
            pnl_pct = trade.get("pnl_pct", 0)
            reason = trade.get("exit_reason", "")

            emoji = "✅" if pnl >= 0 else "❌"

            text += f"{emoji} <b>{symbol}</b> ({side})\n"
            text += f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
            if reason:
                text += f"   Reason: {reason}\n"
            text += "\n"

        return text

    def handle_pnl(self) -> str:
        """Handle /pnl command."""
        trades = self._get_trades(100)
        state = self._get_state()

        # Filter today's trades
        today = datetime.now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in trades
            if t.get("timestamp", "").startswith(today)
        ]

        realized_pnl = sum(t.get("pnl", 0) for t in today_trades)
        wins = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in today_trades if t.get("pnl", 0) < 0)

        # Unrealized P&L
        positions = state.get("positions", {})
        unrealized_pnl = sum(
            p.get("unrealized_pnl", 0)
            for p in positions.values()
        )

        total_pnl = realized_pnl + unrealized_pnl
        emoji = "📈" if total_pnl >= 0 else "📉"

        return f"""
<b>{emoji} Today's P&L</b>

Realized: ${realized_pnl:+,.2f}
Unrealized: ${unrealized_pnl:+,.2f}
<b>Total: ${total_pnl:+,.2f}</b>

Trades: {len(today_trades)}
Wins: {wins} | Losses: {losses}
"""

    def handle_stop(self) -> str:
        """Handle /stop command - pause trading."""
        control = self._get_control()
        control["paused"] = True
        control["paused_at"] = datetime.now().isoformat()
        control["reason"] = "Manual stop via Telegram"
        self._save_control(control)

        return "🔴 <b>Trading PAUSED</b>\n\nUse /start to resume."

    def handle_start(self) -> str:
        """Handle /start command - resume trading."""
        control = self._get_control()
        control["paused"] = False
        control["resumed_at"] = datetime.now().isoformat()
        self._save_control(control)

        return "🟢 <b>Trading RESUMED</b>\n\nBot is now active."

    def handle_help(self) -> str:
        """Handle /help command."""
        return """
<b>🤖 Trading Bot Commands</b>

/status - Bot status overview
/balance - Check balance & P&L
/positions - List open positions
/trades - Recent trades
/pnl - Today's P&L summary

<b>Controls:</b>
/stop - Pause trading
/start - Resume trading

<b>Settings:</b>
/risk - View risk settings
/symbols - Active trading pairs
"""

    def handle_symbols(self) -> str:
        """Handle /symbols command."""
        # Try to get from running bot config
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

        state = self._get_state()
        if "symbols" in state:
            symbols = state["symbols"]

        text = "<b>📋 Active Symbols</b>\n\n"
        for i, sym in enumerate(symbols, 1):
            text += f"{i}. {sym}\n"

        return text

    def handle_risk(self) -> str:
        """Handle /risk command."""
        control = self._get_control()

        shorting = "✅" if control.get("allow_shorting") else "❌"
        leverage = "✅" if control.get("allow_leverage") else "❌"
        aggressive = "✅" if control.get("aggressive_mode") else "❌"

        return f"""
<b>⚠️ Risk Settings</b>

Shorting: {shorting}
Leverage: {leverage}
Aggressive: {aggressive}
Max Leverage: {control.get('max_leverage', 1.0)}x
"""

    def process_command(self, text: str) -> str:
        """Process incoming command."""
        text = text.strip().lower()

        commands = {
            "/status": self.handle_status,
            "/balance": self.handle_balance,
            "/positions": self.handle_positions,
            "/trades": self.handle_trades,
            "/pnl": self.handle_pnl,
            "/stop": self.handle_stop,
            "/start": self.handle_start,
            "/help": self.handle_help,
            "/symbols": self.handle_symbols,
            "/risk": self.handle_risk,
        }

        # Check exact match
        if text in commands:
            return commands[text]()

        # Check prefix match
        for cmd, handler in commands.items():
            if text.startswith(cmd):
                return handler()

        return self.handle_help()

    async def poll_updates(self):
        """Poll for Telegram updates."""
        if not self.token:
            logger.error("Telegram token not configured")
            return

        import aiohttp
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"

        logger.info("Starting Telegram command bot polling...")
        self._running = True

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    params = {
                        "offset": self._last_update_id + 1,
                        "timeout": 30,
                    }

                    async with session.get(url, params=params, timeout=35) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(5)
                            continue

                        data = await resp.json()

                        for update in data.get("result", []):
                            self._last_update_id = update["update_id"]

                            message = update.get("message", {})
                            text = message.get("text", "")
                            chat_id = message.get("chat", {}).get("id")

                            if text.startswith("/"):
                                logger.info(f"Command received: {text}")
                                response = self.process_command(text)

                                # Send response
                                send_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                                await session.post(send_url, json={
                                    "chat_id": chat_id,
                                    "text": response,
                                    "parse_mode": "HTML",
                                })

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Telegram poll error: {e}")
                    await asyncio.sleep(5)

    def stop(self):
        """Stop polling."""
        self._running = False


async def run_telegram_bot():
    """Run the Telegram command bot."""
    from dotenv import load_dotenv
    load_dotenv()

    bot = TelegramCommandBot()
    await bot.poll_updates()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_telegram_bot())
