#!/usr/bin/env python3
"""
Telegram Bot Setup Helper.

Helps you configure Telegram notifications for the trading bot.
"""

import os
import sys
from pathlib import Path


def get_bot_token():
    """Get bot token from user."""
    print("\n" + "=" * 60)
    print("TELEGRAM BOT SETUP")
    print("=" * 60)
    print("""
To receive trading alerts on Telegram, you need:

1. A Telegram Bot Token
2. Your Chat ID

STEP 1: Create a Bot
--------------------
1. Open Telegram and search for @BotFather
2. Send /newbot command
3. Follow the instructions to create your bot
4. Copy the token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
""")

    token = input("Enter your Bot Token: ").strip()
    if not token:
        print("Error: Bot token is required")
        return None

    return token


def get_chat_id(token: str):
    """Help user get their chat ID."""
    print("""
STEP 2: Get Your Chat ID
------------------------
1. Open Telegram and search for your bot (the one you just created)
2. Send any message to your bot (e.g., "hello")
3. Press Enter here to check for your chat ID...
""")

    input("Press Enter after sending a message to your bot...")

    # Try to get updates from the bot
    import json
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        req = Request(url, headers={"Content-Type": "application/json"})

        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get("ok") and data.get("result"):
            for update in data["result"]:
                if "message" in update:
                    chat_id = update["message"]["chat"]["id"]
                    username = update["message"]["from"].get("username", "Unknown")
                    first_name = update["message"]["from"].get("first_name", "Unknown")
                    print(f"\nFound chat from: {first_name} (@{username})")
                    print(f"Chat ID: {chat_id}")
                    return str(chat_id)

        print("\nNo messages found. Make sure you:")
        print("1. Started a conversation with your bot")
        print("2. Sent at least one message")
        print("\nYou can also get your Chat ID from @userinfobot or @getidsbot")

        chat_id = input("\nEnter your Chat ID manually: ").strip()
        return chat_id if chat_id else None

    except URLError as e:
        print(f"\nError connecting to Telegram: {e}")
        chat_id = input("Enter your Chat ID manually: ").strip()
        return chat_id if chat_id else None


def test_connection(token: str, chat_id: str):
    """Test the Telegram connection."""
    print("\nTesting connection...")

    import json
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({
            "chat_id": chat_id,
            "text": "🤖 *Algo Trading Bot Connected!*\n\nYou will now receive trading alerts here.\n\n_This is a test message._",
            "parse_mode": "Markdown",
        }).encode("utf-8")

        req = Request(url, data=data, headers={"Content-Type": "application/json"})

        with urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())

        if result.get("ok"):
            print("✅ Test message sent successfully!")
            return True
        else:
            print(f"❌ Error: {result.get('description', 'Unknown error')}")
            return False

    except URLError as e:
        print(f"❌ Connection error: {e}")
        return False


def save_config(token: str, chat_id: str):
    """Save configuration to .env file."""
    env_path = Path(".env")

    # Read existing .env if it exists
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                # Skip existing Telegram config
                if not line.startswith("TELEGRAM_"):
                    existing_lines.append(line)

    # Add Telegram config
    existing_lines.append(f"\n# Telegram Notifications\n")
    existing_lines.append(f"TELEGRAM_BOT_TOKEN={token}\n")
    existing_lines.append(f"TELEGRAM_CHAT_ID={chat_id}\n")

    with open(env_path, "w") as f:
        f.writelines(existing_lines)

    print(f"\n✅ Configuration saved to {env_path}")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           TELEGRAM NOTIFICATION SETUP                     ║
║                                                           ║
║  This script will help you configure Telegram alerts      ║
║  for the Algo Trading Bot.                                ║
╚══════════════════════════════════════════════════════════╝
""")

    # Check if already configured
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        print("Telegram is already configured!")
        print(f"Bot Token: {os.getenv('TELEGRAM_BOT_TOKEN')[:20]}...")
        print(f"Chat ID: {os.getenv('TELEGRAM_CHAT_ID')}")

        reconfigure = input("\nReconfigure? (y/N): ").strip().lower()
        if reconfigure != 'y':
            print("Exiting.")
            return

    # Get bot token
    token = get_bot_token()
    if not token:
        print("Setup cancelled.")
        return

    # Get chat ID
    chat_id = get_chat_id(token)
    if not chat_id:
        print("Setup cancelled.")
        return

    # Test connection
    if test_connection(token, chat_id):
        # Save configuration
        save = input("\nSave configuration to .env? (Y/n): ").strip().lower()
        if save != 'n':
            save_config(token, chat_id)

        print("""
╔══════════════════════════════════════════════════════════╗
║              SETUP COMPLETE!                              ║
╚══════════════════════════════════════════════════════════╝

To start the paper trading bot with Telegram alerts:

  python run_live_paper_trading.py

Or set environment variables manually:

  export TELEGRAM_BOT_TOKEN='{token}'
  export TELEGRAM_CHAT_ID='{chat_id}'

You will receive alerts for:
  - Bot start/stop
  - Buy signals (LONG)
  - Sell signals (SHORT)
  - Trade executions
  - Portfolio rebalancing
""".format(token=token[:20] + "...", chat_id=chat_id))
    else:
        print("\n❌ Connection test failed. Please check your credentials.")


if __name__ == "__main__":
    main()
