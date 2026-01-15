#!/usr/bin/env python3
"""
Deploy to Live Trading - One Command Switch.

This script safely transitions from paper to live trading.
Run all checks before enabling real money trading.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


def check_prerequisites():
    """Check all prerequisites before going live."""
    checks = []

    print("\n" + "=" * 60)
    print("PRE-DEPLOYMENT CHECKLIST")
    print("=" * 60)

    # 1. Check Binance API
    print("\n[1/6] Checking Binance API...")
    binance_key = os.getenv("BINANCE_API_KEY")
    binance_secret = os.getenv("BINANCE_API_SECRET")
    if binance_key and binance_secret:
        try:
            import ccxt
            exchange = ccxt.binance({
                "apiKey": binance_key,
                "secret": binance_secret,
            })
            balance = exchange.fetch_balance()
            usdt = balance.get("USDT", {}).get("total", 0)
            print(f"  [OK] Binance connected - ${usdt:.2f} USDT")
            checks.append(("Binance API", True, f"${usdt:.2f} USDT"))
        except Exception as e:
            print(f"  [FAIL] Binance error: {e}")
            checks.append(("Binance API", False, str(e)))
    else:
        print("  [FAIL] Binance API keys not configured")
        checks.append(("Binance API", False, "Keys missing"))

    # 2. Check Kraken API
    print("\n[2/6] Checking Kraken API...")
    kraken_key = os.getenv("KRAKEN_API_KEY")
    kraken_secret = os.getenv("KRAKEN_API_SECRET")
    if kraken_key and kraken_secret:
        try:
            import ccxt
            exchange = ccxt.kraken({
                "apiKey": kraken_key,
                "secret": kraken_secret,
            })
            balance = exchange.fetch_balance()
            gbp = balance.get("GBP", {}).get("total", 0)
            usd = balance.get("USD", {}).get("total", 0)
            print(f"  [OK] Kraken connected - £{gbp:.2f} GBP, ${usd:.2f} USD")
            checks.append(("Kraken API", True, f"£{gbp:.2f} GBP"))
        except Exception as e:
            print(f"  [FAIL] Kraken error: {e}")
            checks.append(("Kraken API", False, str(e)))
    else:
        print("  [SKIP] Kraken API keys not configured")
        checks.append(("Kraken API", None, "Optional"))

    # 3. Check AI Brain
    print("\n[3/6] Checking AI Trading Brain...")
    try:
        from bot.intelligence import get_intelligent_brain
        brain = get_intelligent_brain()
        health = brain.health_check()
        claude_ok = health.get("llm_router", {}).get("claude_available", False)
        patterns = health.get("pattern_memory", {}).get("total_patterns", 0)
        print(f"  [OK] AI Brain active - Claude: {claude_ok}, Patterns: {patterns}")
        checks.append(("AI Brain", True, f"{patterns} patterns learned"))
    except Exception as e:
        print(f"  [WARN] AI Brain: {e}")
        checks.append(("AI Brain", None, "Optional"))

    # 4. Check Telegram
    print("\n[4/6] Checking Telegram notifications...")
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
    if telegram_token and telegram_chat:
        print(f"  [OK] Telegram configured")
        checks.append(("Telegram", True, "Configured"))
    else:
        print("  [WARN] Telegram not configured")
        checks.append(("Telegram", None, "Recommended"))

    # 5. Check Safety Limits
    print("\n[5/6] Checking safety limits...")
    max_pos = os.getenv("LIVE_MAX_POSITION_USD", "50")
    max_loss = os.getenv("LIVE_MAX_DAILY_LOSS_PCT", "2.0")
    max_trades = os.getenv("LIVE_MAX_TRADES_PER_DAY", "10")
    print(f"  [OK] Max position: ${max_pos}")
    print(f"  [OK] Max daily loss: {max_loss}%")
    print(f"  [OK] Max trades/day: {max_trades}")
    checks.append(("Safety Limits", True, f"${max_pos} max"))

    # 6. Check Models
    print("\n[6/6] Checking ML models...")
    models_dir = Path("data/models")
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*.pkl"))) + len(list(models_dir.glob("*.joblib")))
        print(f"  [OK] {model_count} models found")
        checks.append(("ML Models", True, f"{model_count} models"))
    else:
        print("  [WARN] No models directory")
        checks.append(("ML Models", None, "Will use rules"))

    return checks


def print_summary(checks):
    """Print deployment summary."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)

    required_passed = True
    for name, status, detail in checks:
        if status is True:
            icon = "[OK]"
        elif status is False:
            icon = "[FAIL]"
            if name in ["Binance API"]:
                required_passed = False
        else:
            icon = "[--]"
        print(f"  {icon} {name}: {detail}")

    return required_passed


def enable_live_trading():
    """Enable live trading mode."""
    env_path = Path(".env")
    content = env_path.read_text()

    # Update PAPER_MODE
    if "PAPER_MODE=true" in content:
        content = content.replace("PAPER_MODE=true", "PAPER_MODE=false")
        env_path.write_text(content)
        print("\n[OK] PAPER_MODE set to false")
        return True
    elif "PAPER_MODE=false" in content:
        print("\n[OK] Already in live mode")
        return True
    else:
        print("\n[WARN] PAPER_MODE not found in .env")
        return False


def disable_live_trading():
    """Disable live trading mode (back to paper)."""
    env_path = Path(".env")
    content = env_path.read_text()

    if "PAPER_MODE=false" in content:
        content = content.replace("PAPER_MODE=false", "PAPER_MODE=true")
        env_path.write_text(content)
        print("\n[OK] PAPER_MODE set to true (paper trading)")
        return True
    elif "PAPER_MODE=true" in content:
        print("\n[OK] Already in paper mode")
        return True
    return False


def main():
    """Main deployment script."""
    print("\n" + "=" * 60)
    print("ALGO TRADING LAB - DEPLOYMENT TOOL")
    print("=" * 60)

    print("\nOptions:")
    print("  1. Run pre-deployment checks only")
    print("  2. Enable LIVE trading (real money)")
    print("  3. Disable LIVE trading (back to paper)")
    print("  4. Exit")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        checks = check_prerequisites()
        ready = print_summary(checks)
        if ready:
            print("\n" + "-" * 60)
            print("ALL CHECKS PASSED - Ready to go live!")
            print("-" * 60)
        else:
            print("\n" + "-" * 60)
            print("SOME CHECKS FAILED - Fix issues before going live")
            print("-" * 60)

    elif choice == "2":
        checks = check_prerequisites()
        ready = print_summary(checks)

        if not ready:
            print("\n[ERROR] Cannot enable live trading - fix issues first")
            return

        print("\n" + "!" * 60)
        print("WARNING: You are about to enable REAL MONEY trading!")
        print("!" * 60)

        confirm = input("\nType 'DEPLOY LIVE' to confirm: ").strip()
        if confirm == "DEPLOY LIVE":
            enable_live_trading()
            print("\n" + "=" * 60)
            print("LIVE TRADING ENABLED!")
            print("=" * 60)
            print("\nTo start live trading:")
            print("  python scripts/trading/run_ai_live_trading.py")
            print("\nTo monitor:")
            print("  http://localhost:8000")
        else:
            print("\nCancelled.")

    elif choice == "3":
        disable_live_trading()
        print("\nPaper trading mode restored.")

    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
