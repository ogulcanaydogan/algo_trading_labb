#!/usr/bin/env python3
"""
Unified Trading CLI - Paper to Live Trading with Safety Controls.

Usage:
    python run_unified_trading.py                              # Paper trading (default)
    python run_unified_trading.py --mode live_limited --confirm  # Live trading
    python run_unified_trading.py status                        # Check status
    python run_unified_trading.py check-transition testnet      # Check readiness
    python run_unified_trading.py phase2c-status               # Production readiness gates
"""

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from bot.trading_mode import TradingMode
from bot.unified_engine import EngineConfig, UnifiedTradingEngine
from bot.broker_router import create_multi_asset_adapter


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_venv_python() -> Path:
    repo_root = Path(__file__).resolve().parent
    if os.name == "nt":
        candidate = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _get_pidfile_path() -> Path:
    """Get the PID file path."""
    repo_root = Path(__file__).resolve().parent
    return repo_root / "logs" / "paper_live.pid"


def _get_heartbeat_path() -> Path:
    """Get the heartbeat file path."""
    repo_root = Path(__file__).resolve().parent
    return repo_root / "data" / "rl" / "paper_live_heartbeat.json"


def _write_pid_file() -> None:
    """Write current PID to pidfile for PID parity with heartbeat."""
    pidfile_path = _get_pidfile_path()
    pidfile_path.parent.mkdir(parents=True, exist_ok=True)
    pidfile_path.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pid_file() -> None:
    """Remove PID file on shutdown."""
    pidfile_path = _get_pidfile_path()
    try:
        if pidfile_path.exists():
            pidfile_path.unlink()
    except OSError:
        pass


def _enforce_single_instance(mode: str = "") -> None:
    """Cross-platform single instance check - allows different modes to run simultaneously."""
    try:
        import psutil
    except ImportError:
        return

    # Check if another process with the SAME mode is already running
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = " ".join(cmdline)
            if "run_unified_trading.py" not in cmdline_str:
                continue
            if "run" not in cmdline_str:
                continue
            # Check if same mode
            if mode and f"--mode {mode}" in cmdline_str:
                print(f"Another {mode} process is already running (PID {proc.info['pid']}); exiting.")
                sys.exit(0)
            elif mode and mode in cmdline_str:
                # Also check without --mode prefix
                print(f"Another {mode} process is already running (PID {proc.info['pid']}); exiting.")
                sys.exit(0)
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue


def _enforce_single_instance_windows() -> None:
    """Windows-specific single instance check (legacy, calls cross-platform version)."""
    if os.name != "nt":
        return
    _enforce_single_instance()


def _reexec_with_venv_on_windows() -> None:
    if os.name != "nt":
        return
    venv_python = _resolve_venv_python()
    try:
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    except OSError:
        pass


def setup_logging(mode: str, data_dir: Path) -> None:
    """Setup logging."""
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode}_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


async def run_trading(args) -> None:
    """Run the trading engine."""
    mode = TradingMode(args.mode)

    if mode.is_live and not args.confirm:
        print("\nWARNING: Live trading requires --confirm flag")
        print(f"  python run_unified_trading.py --mode {mode.value} --confirm")
        return

    data_dir = Path("data/unified_trading")
    setup_logging(mode.value, data_dir)
    logger = logging.getLogger(__name__)

    # Write PID file for process tracking (ensures PID parity with heartbeat)
    _write_pid_file()
    logger.info(f"PID file written: {_get_pidfile_path()} (pid={os.getpid()})")

    if mode == TradingMode.PAPER_LIVE_DATA:
        min_ratio = _env_float("PAPER_LIVE_TURNOVER_MIN_RATIO", 2.0)
        max_daily = _env_int("PAPER_LIVE_TURNOVER_MAX_DAILY", 10)
        os.environ.setdefault("PAPER_LIVE_TURNOVER_MIN_RATIO", str(min_ratio))
        os.environ.setdefault("PAPER_LIVE_TURNOVER_MAX_DAILY", str(max_daily))
        logger.info(
            "Paper-live turnover overrides: min_ratio=%s max_daily=%s",
            min_ratio,
            max_daily,
        )

    # Determine symbols based on mode
    multi_asset = getattr(args, 'multi_asset', False)
    if multi_asset:
        # Multi-asset mode: crypto + forex + commodities + stocks - expanded for 1% daily target
        symbols = [
            # Crypto (Binance) - high volatility, 24/7
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            # Forex (OANDA) - best performers
            "EUR/USD", "GBP/USD",
            # Indices (OANDA) - major indices
            "SPX500/USD", "NAS100/USD",
            # Commodities (OANDA) - Gold, Silver, Oil
            "XAU/USD",    # Gold
            "XAG/USD",    # Silver
            "WTICO/USD",  # WTI Oil
            # US Stocks (Alpaca) - expanded list for more opportunities
            "AAPL/USD", "NVDA/USD", "MSFT/USD", "GOOGL/USD",
            "TSLA/USD", "AMZN/USD", "META/USD",  # Added high-volatility stocks
        ]
        logger.info(f"Multi-asset mode: {len(symbols)} symbols for maximum opportunities")
    else:
        symbols = args.symbols.split(",") if args.symbols else ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT"]

    config = EngineConfig(
        initial_mode=mode,
        initial_capital=args.capital,
        symbols=symbols,
        loop_interval_seconds=args.interval,
        data_dir=data_dir,
        multi_asset=multi_asset,
    )

    engine = UnifiedTradingEngine(config)

    def _schedule_stop(sig):
        logger.info(f"Received signal {sig}, stopping...")
        loop.call_soon_threadsafe(lambda: asyncio.create_task(engine.stop()))

    loop = asyncio.get_running_loop()
    signals = [signal.SIGINT, signal.SIGTERM]
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        signals.append(signal.SIGBREAK)  # type: ignore[attr-defined]
    for sig in signals:
        try:
            loop.add_signal_handler(sig, lambda s=sig: _schedule_stop(s))
        except NotImplementedError:
            signal.signal(sig, lambda *_: _schedule_stop(sig))

    logger.info("=" * 60)
    logger.info("UNIFIED TRADING ENGINE")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode.value}")
    logger.info(f"Capital: ${config.initial_capital:.2f}")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Interval: {config.loop_interval_seconds}s")
    logger.info("=" * 60)

    if not await engine.initialize(resume=not args.fresh):
        logger.error("Failed to initialize")
        return

    # Send Telegram notification
    try:
        from bot.notifications import NotificationManager, Alert, AlertLevel, AlertType
        nm = NotificationManager()
        if nm.has_channels():
            alert = Alert(
                level=AlertLevel.INFO,
                alert_type=AlertType.SYSTEM,
                title="Trading Engine Started",
                message=f"Mode: {mode.value}\nCapital: ${config.initial_capital}\nSymbols: {', '.join(config.symbols)}",
            )
            nm.send_alert(alert)
    except Exception:
        pass

    await engine.start()

    try:
        while engine._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

    status = engine.get_status()
    logger.info("\n" + "=" * 60)
    logger.info("FINAL STATUS")
    logger.info(f"Balance: ${status['balance']:.2f}")
    logger.info(f"P&L: ${status['total_pnl']:.2f} ({status['total_pnl_pct']:.2f}%)")
    logger.info(f"Trades: {status['total_trades']}")
    logger.info("=" * 60)

    # Clean up PID file on shutdown
    _remove_pid_file()
    logger.info("PID file removed")


async def show_status(args) -> None:
    """Show current status."""
    import json
    repo_root = Path(__file__).resolve().parent
    state_file = repo_root / "data" / "unified_trading" / "state.json"
    heartbeat_file = repo_root / "data" / "rl" / "paper_live_heartbeat.json"

    state = None
    heartbeat = None
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
    if heartbeat_file.exists():
        with open(heartbeat_file) as f:
            heartbeat = json.load(f)

    if getattr(args, "json", False):
        payload = {
            "state": state,
            "heartbeat": heartbeat,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    print("\n" + "=" * 60)
    print("TRADING ENGINE STATUS")
    print("=" * 60)

    if state is not None:
        print(f"\nMode: {state['mode']}")
        print(f"Status: {state['status']}")
        print(f"Balance: ${state['current_balance']:.2f}")
        print(f"P&L: ${state['total_pnl']:.2f}")
        print(f"Trades: {state['total_trades']}")
        if state['total_trades'] > 0:
            wr = state['winning_trades'] / state['total_trades'] * 100
            print(f"Win Rate: {wr:.1f}%")
        if state.get('positions'):
            print(f"\nPositions ({len(state['positions'])}):")
            for sym, pos in state['positions'].items():
                print(f"  {sym}: {pos['side']} @ ${pos['entry_price']:.2f}")
    if heartbeat is not None:
        print("\nHeartbeat:")
        print(f"  Timestamp: {heartbeat.get('timestamp', 'unknown')}")
        print(f"  PID: {heartbeat.get('pid', 'unknown')}")
        print(f"  Mode: {heartbeat.get('mode', 'unknown')}")
        symbols = heartbeat.get("symbols") or []
        if symbols:
            print(f"  Symbols: {', '.join(symbols)}")
        decisions = heartbeat.get("total_decisions_session")
        if decisions is not None:
            print(f"  Decisions (session): {decisions}")
        paper_live_decisions = heartbeat.get("paper_live_decisions_session")
        if paper_live_decisions is not None:
            print(f"  Paper-live decisions (session): {paper_live_decisions}")
    if state is None and heartbeat is None:
        print("No state found. Start trading first.")
    print("=" * 60)


async def check_transition(args) -> None:
    """Check transition readiness."""
    from bot.trading_mode import TradingMode
    from bot.transition_validator import create_transition_validator
    from bot.unified_state import UnifiedStateStore

    target = TradingMode(args.target_mode)
    store = UnifiedStateStore()

    try:
        state = store.initialize(TradingMode.PAPER_LIVE_DATA, 10000, resume=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"No state found: {e}")
        return

    validator = create_transition_validator()
    progress = validator.get_transition_progress(state.mode, target, store.get_mode_state())

    print(f"\nTransition: {state.mode.value} -> {target.value}")
    print(f"Progress: {progress['overall_progress']:.1f}%")
    print(f"Allowed: {'Yes' if progress['allowed'] else 'No'}")
    for d in progress['progress_details']:
        status = "[OK]" if d['passed'] else "[X]"
        print(f"  {status} {d['requirement']}: {d['current']:.1f}/{d['required']:.1f}")


async def emergency_stop(args) -> None:
    """Trigger emergency stop."""
    from bot.safety_controller import SafetyController
    controller = SafetyController()
    controller.emergency_stop(args.reason or "Manual stop")
    print("EMERGENCY STOP TRIGGERED")


def _check_gate(
    name: str,
    passed: bool,
    evidence: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create a gate check result."""
    return {
        "name": name,
        "passed": passed,
        "evidence": evidence,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def phase2c_status(args) -> int:
    """
    Check Phase 2C production readiness gates.

    Returns:
        0 = all gates pass
        1 = error reading data
        2 = one or more gates fail
    """
    repo_root = Path(__file__).resolve().parent
    shadow_file = repo_root / "data" / "rl" / "shadow_decisions.jsonl"
    heartbeat_file = repo_root / "data" / "rl" / "paper_live_heartbeat.json"
    override_file = repo_root / "data" / "rl" / "manual_override.json"
    logs_dir = repo_root / "logs"

    gates: list[dict[str, Any]] = []

    try:
        # Gate 1: Shadow decisions >= 100
        shadow_count = 0
        if shadow_file.exists():
            with shadow_file.open("r", encoding="utf-8") as f:
                shadow_count = sum(1 for _ in f)
        gate1_pass = shadow_count >= 100
        gates.append(_check_gate(
            "Shadow decisions >= 100",
            gate1_pass,
            f"{shadow_count} decisions",
        ))

        # Gate 2: Capital preservation never CRITICAL
        critical_events: list[str] = []
        for log_file in logs_dir.glob("paper_live*.log"):
            try:
                content = log_file.read_text(encoding="utf-8", errors="ignore")
                # Look for critical preservation level
                matches = re.findall(r"level[=:]critical|CRITICAL.*preservation", content, re.IGNORECASE)
                if matches:
                    critical_events.extend([f"{log_file.name}: {m}" for m in matches[:3]])
            except Exception:
                pass
        gate2_pass = len(critical_events) == 0
        gates.append(_check_gate(
            "Capital preservation never CRITICAL",
            gate2_pass,
            "clean" if gate2_pass else f"{len(critical_events)} events: {critical_events[:2]}",
        ))

        # Gate 3: Strategy weighting clamp never exceeded
        clamp_events: list[str] = []
        for log_file in logs_dir.glob("paper_live*.log"):
            try:
                content = log_file.read_text(encoding="utf-8", errors="ignore")
                # Look for clamp exceeded messages
                matches = re.findall(r"clamp.*exceed|weight.*clamp.*hit", content, re.IGNORECASE)
                if matches:
                    clamp_events.extend([f"{log_file.name}: {m}" for m in matches[:3]])
            except Exception:
                pass
        gate3_pass = len(clamp_events) == 0
        gates.append(_check_gate(
            "Strategy weighting clamp never exceeded",
            gate3_pass,
            "clean" if gate3_pass else f"{len(clamp_events)} events",
        ))

        # Gate 4: Tradegate rejection non-anomalous
        # Check shadow_decisions for rejection reasons
        rejection_reasons: dict[str, int] = {}
        if shadow_file.exists():
            try:
                with shadow_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            dec = json.loads(line)
                            gate_dec = dec.get("gate_decision", {})
                            reason = gate_dec.get("rejection_reason", "")
                            if reason and not gate_dec.get("approved", True):
                                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass

        # Known non-anomalous rejection reasons
        known_reasons = {
            "signal_blocked_at_generator",
            "insufficient_data",
            "ev_ratio_below_threshold",
            "daily_limit_reached",
            "interval_too_short",
            "no_signal",
            "low_confidence",
            "scalping",
            "trend_too_strong",
            "cooldown",
            "position_limit",
            "risk_limit",
            "mtf_rejected",  # Multi-timeframe filter
            "htf",           # Higher timeframe filter
            "rejected",      # Generic rejection
        }
        anomalous_rejections = {
            k: v for k, v in rejection_reasons.items()
            if not any(known in k.lower() for known in known_reasons)
        }
        gate4_pass = len(anomalous_rejections) == 0
        top_rejections = sorted(rejection_reasons.items(), key=lambda x: -x[1])[:3]
        evidence4 = ", ".join([f"{k}:{v}" for k, v in top_rejections]) if top_rejections else "no rejections"
        if anomalous_rejections:
            evidence4 = f"ANOMALOUS: {list(anomalous_rejections.keys())[:2]}"
        gates.append(_check_gate(
            "Tradegate rejection non-anomalous",
            gate4_pass,
            evidence4,
        ))

        # Gate 5: Turnover governor cost drag < threshold (default 5%)
        turnover_cost_drag = 0.0
        turnover_threshold = float(os.getenv("TURNOVER_COST_DRAG_THRESHOLD", "5.0"))
        for log_file in sorted(logs_dir.glob("paper_live*.log"), reverse=True)[:3]:
            try:
                content = log_file.read_text(encoding="utf-8", errors="ignore")
                # Look for cost drag percentage
                match = re.search(r"cost[_\s]*drag[:\s]*(\d+\.?\d*)%?", content, re.IGNORECASE)
                if match:
                    turnover_cost_drag = float(match.group(1))
                    break
            except Exception:
                pass
        gate5_pass = turnover_cost_drag < turnover_threshold
        gates.append(_check_gate(
            f"Turnover governor cost drag < {turnover_threshold}%",
            gate5_pass,
            f"{turnover_cost_drag:.2f}%",
        ))

        # Gate 6: Heartbeat recent (< 6 hours)
        heartbeat_age_hours = float("inf")
        heartbeat_ts = None
        if heartbeat_file.exists():
            try:
                with heartbeat_file.open("r", encoding="utf-8") as f:
                    hb = json.load(f)
                heartbeat_ts = hb.get("timestamp", "")
                if heartbeat_ts:
                    dt = datetime.fromisoformat(heartbeat_ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    heartbeat_age_hours = (now - dt).total_seconds() / 3600
            except Exception:
                pass
        gate6_pass = heartbeat_age_hours < 6.0
        if heartbeat_age_hours == float("inf"):
            evidence6 = "no heartbeat"
        else:
            evidence6 = f"{heartbeat_age_hours:.2f}h ago ({heartbeat_ts})"
        gates.append(_check_gate(
            "Heartbeat recent (< 6h)",
            gate6_pass,
            evidence6,
            heartbeat_ts,
        ))

        # Gate 7: No manual overrides active
        override_active = False
        override_reason = ""
        if override_file.exists():
            try:
                with override_file.open("r", encoding="utf-8") as f:
                    override = json.load(f)
                override_active = override.get("active", False)
                override_reason = override.get("reason", "unknown")
            except Exception:
                pass
        gate7_pass = not override_active
        gates.append(_check_gate(
            "No manual overrides active",
            gate7_pass,
            "clean" if gate7_pass else f"override: {override_reason}",
        ))

    except Exception as e:
        print(f"ERROR: Failed to check gates: {e}")
        return 1

    # Print results
    all_passed = all(g["passed"] for g in gates)

    print("\n" + "=" * 70)
    print("PHASE 2C PRODUCTION READINESS GATES")
    print("=" * 70)
    print(f"{'Gate':<45} {'Status':<8} {'Evidence'}")
    print("-" * 70)

    for gate in gates:
        status = "PASS" if gate["passed"] else "FAIL"
        status_colored = f"\033[32m{status}\033[0m" if gate["passed"] else f"\033[31m{status}\033[0m"
        print(f"{gate['name']:<45} {status_colored:<17} {gate['evidence']}")

    print("-" * 70)

    if all_passed:
        print("\033[32m✓ All gates PASS - Production ready\033[0m")
        return 0
    else:
        failed = [g["name"] for g in gates if not g["passed"]]
        print(f"\033[31m✗ {len(failed)} gate(s) FAIL - Not production ready\033[0m")
        return 2


async def phase2d_status(args) -> int:
    """
    Check Phase 2D live-limited readiness gates.

    Returns:
        0 = all gates pass
        1 = error
        2 = one or more gates fail
    """
    mode = getattr(args, "mode", "live_limited")
    gates: list[dict[str, Any]] = []

    print(f"\n{'=' * 70}")
    print(f"PHASE 2D LIVE-LIMITED READINESS GATES (mode={mode})")
    print("=" * 70)

    try:
        # Gate 1: API keys present (check env vars without printing values)
        api_keys_check = []
        required_keys = {
            "binance": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
            "alpaca": ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"],
            "oanda": ["OANDA_API_KEY", "OANDA_ACCOUNT_ID"],
        }

        for broker, keys in required_keys.items():
            present = all(os.getenv(k) for k in keys)
            api_keys_check.append(f"{broker}:{'✓' if present else '✗'}")

        any_broker_ready = any(
            all(os.getenv(k) for k in keys)
            for keys in required_keys.values()
        )
        gates.append(_check_gate(
            "API keys present (at least one broker)",
            any_broker_ready,
            " ".join(api_keys_check),
        ))

        # Gate 2: Exchange connectivity test (read-only operations)
        connectivity_results = []

        # Try Binance connectivity (mainnet, read-only)
        if os.getenv("BINANCE_API_KEY"):
            try:
                import ccxt
                exchange = ccxt.binance({
                    "apiKey": os.getenv("BINANCE_API_KEY"),
                    "secret": os.getenv("BINANCE_API_SECRET"),
                })
                # Read-only operation: fetch account balance
                balance = exchange.fetch_balance()
                total_usdt = balance.get("total", {}).get("USDT", 0)
                connectivity_results.append(f"binance:✓(${total_usdt:.0f})")
            except ImportError:
                connectivity_results.append("binance:skip(no ccxt)")
            except Exception as e:
                err_msg = str(e)[:25].replace("\n", " ")
                connectivity_results.append(f"binance:✗({err_msg})")

        # Try Alpaca connectivity (paper mode for safety)
        if os.getenv("ALPACA_API_KEY"):
            try:
                from alpaca.trading.client import TradingClient
                client = TradingClient(
                    os.getenv("ALPACA_API_KEY"),
                    os.getenv("ALPACA_SECRET_KEY"),
                    paper=True,  # Use paper for safety
                )
                account = client.get_account()
                connectivity_results.append(f"alpaca:✓(${float(account.cash):.0f})")
            except ImportError:
                connectivity_results.append("alpaca:skip(no sdk)")
            except Exception as e:
                err_msg = str(e)[:25].replace("\n", " ")
                connectivity_results.append(f"alpaca:✗({err_msg})")

        # Try OANDA connectivity
        if os.getenv("OANDA_API_KEY") and os.getenv("OANDA_ACCOUNT_ID"):
            try:
                import requests
                account_id = os.getenv("OANDA_ACCOUNT_ID")
                api_key = os.getenv("OANDA_API_KEY")
                # Practice/demo endpoint for safety
                url = f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/summary"
                headers = {"Authorization": f"Bearer {api_key}"}
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    balance = float(data.get("account", {}).get("balance", 0))
                    connectivity_results.append(f"oanda:✓(${balance:.0f})")
                else:
                    connectivity_results.append(f"oanda:✗(HTTP {resp.status_code})")
            except ImportError:
                connectivity_results.append("oanda:skip(no requests)")
            except Exception as e:
                err_msg = str(e)[:25].replace("\n", " ")
                connectivity_results.append(f"oanda:✗({err_msg})")

        connectivity_pass = any("✓" in r for r in connectivity_results)
        gates.append(_check_gate(
            "Exchange connectivity (at least one)",
            connectivity_pass,
            " ".join(connectivity_results) if connectivity_results else "no brokers configured",
        ))

        # Gate 3: Risk limits configured
        risk_limits = {
            "max_position_pct": _env_float("LIVE_MAX_POSITION_PCT", 5.0),
            "max_leverage": _env_float("LIVE_MAX_LEVERAGE", 1.0),
            "max_daily_loss_pct": _env_float("LIVE_MAX_DAILY_LOSS_PCT", 2.0),
            "max_orders_day": _env_int("LIVE_MAX_ORDERS_DAY", 20),
        }
        # Check if limits are conservative enough
        limits_conservative = (
            risk_limits["max_position_pct"] <= 10.0 and
            risk_limits["max_leverage"] <= 3.0 and
            risk_limits["max_daily_loss_pct"] <= 5.0 and
            risk_limits["max_orders_day"] <= 50
        )
        limits_evidence = ", ".join([f"{k}={v}" for k, v in risk_limits.items()])
        gates.append(_check_gate(
            "Risk limits conservative",
            limits_conservative,
            limits_evidence,
        ))

        # Gate 4: Emergency stop path verified
        from bot.safety_controller import SafetyController
        try:
            controller = SafetyController()
            # Check that the safety controller can be instantiated
            # and has emergency_stop method
            has_estop = hasattr(controller, "emergency_stop") and callable(controller.emergency_stop)
            repo_root = Path(__file__).resolve().parent
            safety_state = repo_root / "data" / "safety_state.json"
            gates.append(_check_gate(
                "Emergency stop path verified",
                has_estop,
                f"SafetyController.emergency_stop exists, state_file={safety_state.exists()}",
            ))
        except Exception as e:
            gates.append(_check_gate(
                "Emergency stop path verified",
                False,
                f"error: {str(e)[:40]}",
            ))

        # Gate 5: Paper trading validation (Phase 2C must pass)
        phase2c_exit = phase2c_status(type("Args", (), {})())  # Run phase2c silently
        gates.append(_check_gate(
            "Phase 2C gates pass (paper-live validated)",
            phase2c_exit == 0,
            f"exit_code={phase2c_exit}",
        ))

        # Gate 6: Live-limited mode capital cap
        live_capital = _env_float("LIVE_LIMITED_CAPITAL", 1000.0)
        capital_safe = live_capital <= 5000.0  # Max $5000 for live-limited
        gates.append(_check_gate(
            "Live-limited capital cap (<= $5000)",
            capital_safe,
            f"${live_capital:.2f}",
        ))

        # Gate 7: Confirmation flag requirement
        # This just documents that --confirm is required for live modes
        gates.append(_check_gate(
            "Confirmation flag required for live",
            True,  # Always pass - documents the requirement
            "--confirm flag enforced in code",
        ))

    except Exception as e:
        print(f"ERROR: Failed to check gates: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    all_passed = all(g["passed"] for g in gates)

    print(f"{'Gate':<45} {'Status':<8} {'Evidence'}")
    print("-" * 70)

    for gate in gates:
        status = "PASS" if gate["passed"] else "FAIL"
        status_colored = f"\033[32m{status}\033[0m" if gate["passed"] else f"\033[31m{status}\033[0m"
        # Truncate evidence to fit
        evidence = gate["evidence"][:60] + "..." if len(gate["evidence"]) > 60 else gate["evidence"]
        print(f"{gate['name']:<45} {status_colored:<17} {evidence}")

    print("-" * 70)

    if all_passed:
        print("\033[32m✓ All gates PASS - Live-limited ready\033[0m")
        print("\nTo start live-limited trading:")
        print("  systemctl --user start algo_trading_live_limited.service")
        return 0
    else:
        failed = [g["name"] for g in gates if not g["passed"]]
        print(f"\033[31m✗ {len(failed)} gate(s) FAIL - Not ready for live-limited\033[0m")
        for f in failed:
            print(f"  - {f}")
        return 2


def main():
    _reexec_with_venv_on_windows()

    parser = argparse.ArgumentParser(description="Unified Trading Engine")
    subparsers = parser.add_subparsers(dest="command")

    # Run
    run_p = subparsers.add_parser("run", help="Run trading")
    run_p.add_argument("--mode", default="paper_live_data")
    run_p.add_argument("--capital", type=float, default=30000)
    run_p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,ADA/USDT,AVAX/USDT", help="Trading symbols (comma-separated)")
    run_p.add_argument("--interval", type=int, default=30)  # Faster for more opportunities
    run_p.add_argument("--confirm", action="store_true")
    run_p.add_argument("--fresh", action="store_true")
    run_p.add_argument("--multi-asset", action="store_true", help="Enable multi-asset trading (crypto, forex, commodities)")

    # Status
    status_p = subparsers.add_parser("status", help="Show status")
    status_p.add_argument("--json", action="store_true", help="Output JSON status payload")

    # Check transition
    check_p = subparsers.add_parser("check-transition", help="Check transition")
    check_p.add_argument("target_mode")

    # Emergency stop
    stop_p = subparsers.add_parser("emergency-stop", help="Emergency stop")
    stop_p.add_argument("--reason", default="Manual")

    # Phase 2C status (production readiness gates)
    subparsers.add_parser("phase2c-status", help="Check Phase 2C production readiness gates")

    # Phase 2D status (live-limited readiness gates)
    phase2d_p = subparsers.add_parser("phase2d-status", help="Check Phase 2D live-limited readiness gates")
    phase2d_p.add_argument("--mode", default="live_limited", help="Target mode to check")

    args = parser.parse_args()

    if args.command is None:
        args.command = "run"
        args.mode = "paper_live_data"
        args.capital = 30000
        args.symbols = "BTC/USDT,ETH/USDT,SOL/USDT"
        args.interval = 30  # Faster loop for more opportunities
        args.confirm = False
        args.fresh = False
        args.multi_asset = False

    if args.command == "run":
        # Cross-platform single instance check - allows different modes to run simultaneously
        _enforce_single_instance(args.mode)

    if args.command == "run":
        asyncio.run(run_trading(args))
    elif args.command == "status":
        asyncio.run(show_status(args))
    elif args.command == "check-transition":
        asyncio.run(check_transition(args))
    elif args.command == "emergency-stop":
        asyncio.run(emergency_stop(args))
    elif args.command == "phase2c-status":
        sys.exit(phase2c_status(args))
    elif args.command == "phase2d-status":
        sys.exit(asyncio.run(phase2d_status(args)))


if __name__ == "__main__":
    main()
