#!/usr/bin/env python3
"""
Telegram Bot Setup Helper.

This script helps you set up Telegram notifications for your trading bot.

Usage:
    python setup_telegram.py
"""

import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


def get_bot_info(token: str) -> dict:
    """Get bot info from Telegram API."""
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_test_message(token: str, chat_id: str) -> bool:
    """Send a test message."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        message = """
🤖 Trading Bot Connected!

Your trading bot is now connected to Telegram.
You will receive notifications for:
• Trade opened/closed
• Stop loss / Take profit hits
• Daily summaries
• Risk warnings
• System errors

Happy Trading! 📈
        """

        data = json.dumps({
            "chat_id": chat_id,
            "text": message.strip(),
        }).encode("utf-8")

        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            return result.get("ok", False)
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_updates(token: str) -> dict:
    """Get recent updates to find chat_id."""
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def update_env_file(token: str, chat_id: str):
    """Update .env file with Telegram credentials."""
    env_path = Path(__file__).parent / ".env"

    # Read existing content
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r") as f:
            existing_lines = f.readlines()

    # Filter out old Telegram settings
    new_lines = []
    for line in existing_lines:
        if not line.strip().startswith("TELEGRAM_"):
            new_lines.append(line)

    # Add new settings
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines.append("\n")

    new_lines.append(f"\n# Telegram Notifications\n")
    new_lines.append(f"TELEGRAM_BOT_TOKEN={token}\n")
    new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}\n")

    # Write back
    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print(f"\n✅ Updated {env_path}")


def main():
    print("=" * 60)
    print("📱 TELEGRAM BOT SETUP")
    print("=" * 60)
    print()
    print("This will help you set up Telegram notifications for trades.")
    print()

    # Step 1: Create bot
    print("STEP 1: Create a Telegram Bot")
    print("-" * 40)
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot and follow the instructions")
    print("3. BotFather will give you a token like:")
    print("   123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
    print()

    token = input("Enter your bot token: ").strip()

    if not token:
        print("❌ No token provided!")
        return

    # Verify token
    print("\nVerifying token...")
    info = get_bot_info(token)

    if not info.get("ok"):
        print(f"❌ Invalid token! Error: {info.get('error', 'Unknown')}")
        return

    bot_name = info["result"]["username"]
    print(f"✅ Bot verified: @{bot_name}")

    # Step 2: Get chat_id
    print()
    print("STEP 2: Get Your Chat ID")
    print("-" * 40)
    print(f"1. Open Telegram and search for @{bot_name}")
    print("2. Click START or send any message to the bot")
    print("3. Press Enter here after sending a message...")
    input()

    updates = get_updates(token)

    if not updates.get("ok") or not updates.get("result"):
        print("❌ No messages found. Make sure you sent a message to the bot!")
        print()
        chat_id = input("Or enter your chat_id manually (if you know it): ").strip()
        if not chat_id:
            return
    else:
        # Get the most recent chat_id
        chat_id = None
        for update in updates["result"]:
            if "message" in update:
                chat_id = str(update["message"]["chat"]["id"])
                chat_name = update["message"]["chat"].get("first_name", "Unknown")
                break

        if not chat_id:
            print("❌ No chat_id found!")
            return

        print(f"✅ Found chat_id: {chat_id} ({chat_name})")

    # Step 3: Send test message
    print()
    print("STEP 3: Send Test Message")
    print("-" * 40)
    print("Sending test message...")

    if send_test_message(token, chat_id):
        print("✅ Test message sent! Check your Telegram.")
    else:
        print("❌ Failed to send test message.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            return

    # Step 4: Save to .env
    print()
    print("STEP 4: Save Configuration")
    print("-" * 40)

    save = input("Save to .env file? (y/n): ").strip().lower()
    if save == "y":
        update_env_file(token, chat_id)
    else:
        print()
        print("Add these to your .env file manually:")
        print(f"TELEGRAM_BOT_TOKEN={token}")
        print(f"TELEGRAM_CHAT_ID={chat_id}")

    # Done
    print()
    print("=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Your trading bot will now send notifications to Telegram.")
    print()
    print("To test notifications, run:")
    print("  python -c \"from bot.notifications import NotificationManager; m = NotificationManager(); print(m.notify_daily_summary(10000, 100, 1.0, 5, 60))\"")
    print()


if __name__ == "__main__":
    main()
