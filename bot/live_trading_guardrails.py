"""
Micro-Live Rollout Guardrails + Kill Switch.

CRITICAL SAFETY MODULE for transitioning from paper to real trading.

This module enforces strict caps on live trading to prevent catastrophic losses
during the initial rollout phase. ALL guardrails must pass before any real
order is executed.

Features:
- LIVE_MODE toggle (default: OFF)
- Symbol allowlist (only approved symbols can trade)
- Daily trade count limits
- Position size caps
- Leverage limits
- Kill switch (file or env var)
- Persistent state (survives restarts)

IMPORTANT: This is a SAFETY module. Do not bypass these checks.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LiveTradingConfig:
    """
    Configuration for live trading guardrails.

    ALL DEFAULTS ARE SAFE (live trading disabled).
    """

    # Master switch - must be explicitly enabled
    live_mode: bool = False

    # Capital limits
    live_max_capital_pct: float = 0.01  # Max 1% of portfolio in live trades
    live_max_position_pct: float = 0.02  # Max 2% per position

    # Symbol restrictions
    live_symbol_allowlist: List[str] = field(default_factory=lambda: ["ETH/USDT"])

    # Trade limits
    live_max_trades_per_day: int = 3

    # Leverage limits (1.0 = no leverage)
    live_max_leverage: float = 1.0

    # Kill switch - file path or env var name
    kill_switch_file: str = "data/live_kill_switch.txt"
    kill_switch_env_var: str = "LIVE_KILL_SWITCH"

    # State persistence
    state_file: str = "data/live_trading_state.json"

    @classmethod
    def from_env(cls) -> "LiveTradingConfig":
        """Load config from environment variables with safe defaults."""
        allowlist_str = os.getenv("LIVE_SYMBOL_ALLOWLIST", "ETH/USDT")
        allowlist = [s.strip() for s in allowlist_str.split(",") if s.strip()]

        return cls(
            live_mode=os.getenv("LIVE_MODE", "false").lower() == "true",
            live_max_capital_pct=float(os.getenv("LIVE_MAX_CAPITAL_PCT", "0.01")),
            live_max_position_pct=float(os.getenv("LIVE_MAX_POSITION_PCT", "0.02")),
            live_symbol_allowlist=allowlist,
            live_max_trades_per_day=int(os.getenv("LIVE_MAX_TRADES_PER_DAY", "3")),
            live_max_leverage=float(os.getenv("LIVE_MAX_LEVERAGE", "1.0")),
            kill_switch_file=os.getenv("LIVE_KILL_SWITCH_FILE", "data/live_kill_switch.txt"),
            kill_switch_env_var=os.getenv("LIVE_KILL_SWITCH_ENV", "LIVE_KILL_SWITCH"),
            state_file=os.getenv("LIVE_STATE_FILE", "data/live_trading_state.json"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict."""
        return {
            "live_mode": self.live_mode,
            "live_max_capital_pct": self.live_max_capital_pct,
            "live_max_position_pct": self.live_max_position_pct,
            "live_symbol_allowlist": self.live_symbol_allowlist,
            "live_max_trades_per_day": self.live_max_trades_per_day,
            "live_max_leverage": self.live_max_leverage,
            "kill_switch_file": self.kill_switch_file,
            "kill_switch_env_var": self.kill_switch_env_var,
            "state_file": self.state_file,
        }


# =============================================================================
# GUARDRAIL CHECK RESULT
# =============================================================================

@dataclass
class GuardrailCheckResult:
    """Result of a guardrail check."""

    passed: bool
    block_reason: Optional[str] = None
    guardrail_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        """Check if this is a CRITICAL block (kill switch)."""
        return self.guardrail_name == "kill_switch"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "block_reason": self.block_reason,
            "guardrail_name": self.guardrail_name,
            "details": self.details,
            "is_critical": self.is_critical,
        }


# =============================================================================
# PERSISTENT STATE
# =============================================================================

@dataclass
class LiveTradingState:
    """
    Persistent state for live trading guardrails.

    Survives restarts to maintain accurate trade counts.
    """

    # Daily trade tracking
    daily_trade_count: int = 0
    last_trade_date: str = ""  # YYYY-MM-DD
    last_trade_timestamp: Optional[str] = None

    # Trade history (symbol -> count today)
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)

    # Total capital deployed today
    capital_deployed_today: float = 0.0

    # Kill switch history
    kill_switch_activated_at: Optional[str] = None
    kill_switch_reason: Optional[str] = None

    def reset_daily_counters(self):
        """Reset daily counters for a new day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.trades_by_symbol = {}
            self.capital_deployed_today = 0.0
            self.last_trade_date = today

    def record_trade(self, symbol: str, position_value: float):
        """Record a live trade."""
        self.reset_daily_counters()
        self.daily_trade_count += 1
        self.trades_by_symbol[symbol] = self.trades_by_symbol.get(symbol, 0) + 1
        self.capital_deployed_today += position_value
        self.last_trade_timestamp = datetime.now(timezone.utc).isoformat()
        self.last_trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily_trade_count": self.daily_trade_count,
            "last_trade_date": self.last_trade_date,
            "last_trade_timestamp": self.last_trade_timestamp,
            "trades_by_symbol": self.trades_by_symbol,
            "capital_deployed_today": self.capital_deployed_today,
            "kill_switch_activated_at": self.kill_switch_activated_at,
            "kill_switch_reason": self.kill_switch_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveTradingState":
        state = cls(
            daily_trade_count=data.get("daily_trade_count", 0),
            last_trade_date=data.get("last_trade_date", ""),
            last_trade_timestamp=data.get("last_trade_timestamp"),
            trades_by_symbol=data.get("trades_by_symbol", {}),
            capital_deployed_today=data.get("capital_deployed_today", 0.0),
            kill_switch_activated_at=data.get("kill_switch_activated_at"),
            kill_switch_reason=data.get("kill_switch_reason"),
        )
        if not state.last_trade_date:
            state.reset_daily_counters()
        return state


# =============================================================================
# LIVE TRADING GUARDRAILS
# =============================================================================

class LiveTradingGuardrails:
    """
    Enforces strict guardrails on live trading.

    CRITICAL: This is the last line of defense before real money trades.
    All checks must pass before any live order is executed.
    """

    def __init__(
        self,
        config: Optional[LiveTradingConfig] = None,
        state_file_override: Optional[str] = None,
    ):
        self.config = config or LiveTradingConfig.from_env()
        self._state_file = Path(state_file_override or self.config.state_file)
        self._state = self._load_state()
        self._symbol_allowlist: Set[str] = set(
            s.upper().replace("/", "_").replace("-", "_")
            for s in self.config.live_symbol_allowlist
        )

        # Log initialization
        if self.config.live_mode:
            logger.warning(
                "=" * 60 + "\n"
                "LIVE TRADING GUARDRAILS INITIALIZED - LIVE MODE ENABLED\n"
                f"  Max capital: {self.config.live_max_capital_pct * 100:.1f}%\n"
                f"  Max position: {self.config.live_max_position_pct * 100:.1f}%\n"
                f"  Max trades/day: {self.config.live_max_trades_per_day}\n"
                f"  Allowed symbols: {self.config.live_symbol_allowlist}\n"
                f"  Max leverage: {self.config.live_max_leverage}x\n"
                + "=" * 60
            )
        else:
            logger.info(
                "Live trading guardrails initialized - LIVE MODE DISABLED (safe)"
            )

    def _load_state(self) -> LiveTradingState:
        """Load persistent state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                state = LiveTradingState.from_dict(data)
                logger.debug(f"Loaded live trading state: {state.to_dict()}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load live trading state: {e}")

        return LiveTradingState()

    def _save_state(self):
        """Save persistent state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save live trading state: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for comparison."""
        return symbol.upper().replace("/", "_").replace("-", "_")

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    def is_kill_switch_active(self) -> tuple[bool, Optional[str]]:
        """
        Check if kill switch is active.

        Kill switch can be activated by:
        1. File presence (e.g., data/live_kill_switch.txt)
        2. Environment variable set to "true"

        Returns:
            Tuple of (is_active, reason)
        """
        # Check file-based kill switch
        kill_file = Path(self.config.kill_switch_file)
        if kill_file.exists():
            try:
                reason = kill_file.read_text().strip() or "Kill switch file present"
                return True, reason
            except Exception:
                return True, "Kill switch file present (unreadable)"

        # Check env var kill switch
        env_value = os.getenv(self.config.kill_switch_env_var, "").lower()
        if env_value in ("true", "1", "yes", "active"):
            return True, f"Kill switch env var {self.config.kill_switch_env_var} is set"

        return False, None

    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Programmatically activate the kill switch.

        Creates the kill switch file to halt all live trading.
        """
        kill_file = Path(self.config.kill_switch_file)
        kill_file.parent.mkdir(parents=True, exist_ok=True)
        kill_file.write_text(f"{reason}\nActivated at: {datetime.now(timezone.utc).isoformat()}")

        self._state.kill_switch_activated_at = datetime.now(timezone.utc).isoformat()
        self._state.kill_switch_reason = reason
        self._save_state()

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self):
        """
        Deactivate the kill switch.

        Removes the kill switch file to allow live trading to resume.
        """
        kill_file = Path(self.config.kill_switch_file)
        if kill_file.exists():
            kill_file.unlink()
            logger.warning("Kill switch deactivated - live trading can resume")

    # =========================================================================
    # GUARDRAIL CHECKS
    # =========================================================================

    def check_live_mode_enabled(self) -> GuardrailCheckResult:
        """Check if live mode is enabled."""
        if not self.config.live_mode:
            return GuardrailCheckResult(
                passed=False,
                block_reason="LIVE_MODE is disabled",
                guardrail_name="live_mode",
                details={"live_mode": False},
            )
        return GuardrailCheckResult(passed=True, guardrail_name="live_mode")

    def check_kill_switch(self) -> GuardrailCheckResult:
        """Check if kill switch is active."""
        is_active, reason = self.is_kill_switch_active()
        if is_active:
            return GuardrailCheckResult(
                passed=False,
                block_reason=f"KILL SWITCH ACTIVE: {reason}",
                guardrail_name="kill_switch",
                details={"kill_switch_active": True, "reason": reason},
            )
        return GuardrailCheckResult(passed=True, guardrail_name="kill_switch")

    def check_symbol_allowlist(self, symbol: str) -> GuardrailCheckResult:
        """Check if symbol is in the allowlist."""
        normalized = self._normalize_symbol(symbol)
        if normalized not in self._symbol_allowlist:
            return GuardrailCheckResult(
                passed=False,
                block_reason=f"Symbol {symbol} not in live allowlist: {self.config.live_symbol_allowlist}",
                guardrail_name="symbol_allowlist",
                details={
                    "symbol": symbol,
                    "normalized": normalized,
                    "allowlist": list(self._symbol_allowlist),
                },
            )
        return GuardrailCheckResult(passed=True, guardrail_name="symbol_allowlist")

    def check_daily_trade_limit(self) -> GuardrailCheckResult:
        """Check if daily trade limit has been reached."""
        self._state.reset_daily_counters()
        if self._state.daily_trade_count >= self.config.live_max_trades_per_day:
            return GuardrailCheckResult(
                passed=False,
                block_reason=f"Daily trade limit reached: {self._state.daily_trade_count}/{self.config.live_max_trades_per_day}",
                guardrail_name="daily_trade_limit",
                details={
                    "current_count": self._state.daily_trade_count,
                    "max_allowed": self.config.live_max_trades_per_day,
                },
            )
        return GuardrailCheckResult(
            passed=True,
            guardrail_name="daily_trade_limit",
            details={
                "current_count": self._state.daily_trade_count,
                "max_allowed": self.config.live_max_trades_per_day,
            },
        )

    def check_position_size(
        self, position_value: float, portfolio_value: float
    ) -> GuardrailCheckResult:
        """Check if position size is within limits."""
        if portfolio_value <= 0:
            return GuardrailCheckResult(
                passed=False,
                block_reason="Invalid portfolio value",
                guardrail_name="position_size",
                details={"portfolio_value": portfolio_value},
            )

        position_pct = position_value / portfolio_value
        if position_pct > self.config.live_max_position_pct:
            return GuardrailCheckResult(
                passed=False,
                block_reason=(
                    f"Position size {position_pct * 100:.2f}% exceeds max "
                    f"{self.config.live_max_position_pct * 100:.2f}%"
                ),
                guardrail_name="position_size",
                details={
                    "position_value": position_value,
                    "portfolio_value": portfolio_value,
                    "position_pct": position_pct,
                    "max_position_pct": self.config.live_max_position_pct,
                },
            )
        return GuardrailCheckResult(passed=True, guardrail_name="position_size")

    def check_total_capital(
        self, position_value: float, portfolio_value: float
    ) -> GuardrailCheckResult:
        """Check if total capital deployed is within limits."""
        self._state.reset_daily_counters()
        if portfolio_value <= 0:
            return GuardrailCheckResult(
                passed=False,
                block_reason="Invalid portfolio value",
                guardrail_name="total_capital",
            )

        total_deployed = self._state.capital_deployed_today + position_value
        total_pct = total_deployed / portfolio_value

        if total_pct > self.config.live_max_capital_pct:
            return GuardrailCheckResult(
                passed=False,
                block_reason=(
                    f"Total capital {total_pct * 100:.2f}% would exceed max "
                    f"{self.config.live_max_capital_pct * 100:.2f}%"
                ),
                guardrail_name="total_capital",
                details={
                    "current_deployed": self._state.capital_deployed_today,
                    "new_position": position_value,
                    "total_deployed": total_deployed,
                    "portfolio_value": portfolio_value,
                    "total_pct": total_pct,
                    "max_capital_pct": self.config.live_max_capital_pct,
                },
            )
        return GuardrailCheckResult(passed=True, guardrail_name="total_capital")

    def check_leverage(self, leverage: float) -> GuardrailCheckResult:
        """Check if leverage is within limits."""
        if leverage > self.config.live_max_leverage:
            return GuardrailCheckResult(
                passed=False,
                block_reason=f"Leverage {leverage}x exceeds max {self.config.live_max_leverage}x",
                guardrail_name="leverage",
                details={
                    "requested_leverage": leverage,
                    "max_leverage": self.config.live_max_leverage,
                },
            )
        return GuardrailCheckResult(passed=True, guardrail_name="leverage")

    # =========================================================================
    # MAIN GUARDRAIL CHECK
    # =========================================================================

    def check_all_guardrails(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        leverage: float = 1.0,
    ) -> GuardrailCheckResult:
        """
        Run ALL guardrail checks before allowing a live trade.

        Returns the first failed check, or a passed result if all checks pass.

        IMPORTANT: Order of checks matters - kill switch is checked first.
        """
        # 1. Kill switch - MOST CRITICAL
        result = self.check_kill_switch()
        if not result.passed:
            logger.critical(f"LIVE TRADE BLOCKED: {result.block_reason}")
            return result

        # 2. Live mode enabled
        result = self.check_live_mode_enabled()
        if not result.passed:
            logger.info(f"Live trade blocked: {result.block_reason}")
            return result

        # 3. Symbol allowlist
        result = self.check_symbol_allowlist(symbol)
        if not result.passed:
            logger.warning(f"Live trade blocked: {result.block_reason}")
            return result

        # 4. Daily trade limit
        result = self.check_daily_trade_limit()
        if not result.passed:
            logger.warning(f"Live trade blocked: {result.block_reason}")
            return result

        # 5. Position size
        result = self.check_position_size(position_value, portfolio_value)
        if not result.passed:
            logger.warning(f"Live trade blocked: {result.block_reason}")
            return result

        # 6. Total capital
        result = self.check_total_capital(position_value, portfolio_value)
        if not result.passed:
            logger.warning(f"Live trade blocked: {result.block_reason}")
            return result

        # 7. Leverage
        result = self.check_leverage(leverage)
        if not result.passed:
            logger.warning(f"Live trade blocked: {result.block_reason}")
            return result

        # All checks passed
        logger.info(
            f"Live trade APPROVED: {symbol} | "
            f"Position: ${position_value:.2f} | "
            f"Trade #{self._state.daily_trade_count + 1}/{self.config.live_max_trades_per_day}"
        )
        return GuardrailCheckResult(
            passed=True,
            guardrail_name="all",
            details={
                "symbol": symbol,
                "position_value": position_value,
                "portfolio_value": portfolio_value,
                "leverage": leverage,
                "trades_today": self._state.daily_trade_count,
            },
        )

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def record_live_trade(self, symbol: str, position_value: float):
        """
        Record a completed live trade.

        Call this AFTER a live trade is successfully executed.
        """
        self._state.record_trade(symbol, position_value)
        self._save_state()
        logger.info(
            f"Recorded live trade: {symbol} | "
            f"Value: ${position_value:.2f} | "
            f"Total today: {self._state.daily_trade_count}"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current status of live trading guardrails."""
        self._state.reset_daily_counters()
        kill_active, kill_reason = self.is_kill_switch_active()

        return {
            "live_mode_enabled": self.config.live_mode,
            "kill_switch_active": kill_active,
            "kill_switch_reason": kill_reason,
            "daily_trades": {
                "count": self._state.daily_trade_count,
                "limit": self.config.live_max_trades_per_day,
                "remaining": max(0, self.config.live_max_trades_per_day - self._state.daily_trade_count),
            },
            "capital": {
                "deployed_today": self._state.capital_deployed_today,
                "max_pct": self.config.live_max_capital_pct,
            },
            "position": {
                "max_pct": self.config.live_max_position_pct,
            },
            "leverage": {
                "max": self.config.live_max_leverage,
            },
            "symbol_allowlist": self.config.live_symbol_allowlist,
            "last_trade": self._state.last_trade_timestamp,
            "config": self.config.to_dict(),
        }

    def reset_daily_state(self):
        """Manually reset daily state (for testing or admin override)."""
        self._state.daily_trade_count = 0
        self._state.trades_by_symbol = {}
        self._state.capital_deployed_today = 0.0
        self._state.last_trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_state()
        logger.warning("Live trading daily state manually reset")


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_guardrails_instance: Optional[LiveTradingGuardrails] = None


def get_live_guardrails(
    config: Optional[LiveTradingConfig] = None,
) -> LiveTradingGuardrails:
    """Get or create the singleton guardrails instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = LiveTradingGuardrails(config=config)
    return _guardrails_instance


def reset_live_guardrails():
    """Reset the singleton instance (for testing)."""
    global _guardrails_instance
    _guardrails_instance = None


# =============================================================================
# STARTUP SAFETY ASSERTION
# =============================================================================


class StartupReadinessResult:
    """Result of startup readiness check."""

    def __init__(
        self,
        passed: bool,
        live_rollout_readiness: str,
        reasons: List[str],
        kill_switch_activated: bool = False,
    ):
        self.passed = passed
        self.live_rollout_readiness = live_rollout_readiness
        self.reasons = reasons
        self.kill_switch_activated = kill_switch_activated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "live_rollout_readiness": self.live_rollout_readiness,
            "reasons": self.reasons,
            "kill_switch_activated": self.kill_switch_activated,
        }


def check_startup_readiness(
    guardrails: Optional[LiveTradingGuardrails] = None,
    activate_kill_switch_on_failure: bool = True,
) -> StartupReadinessResult:
    """
    Check system readiness at startup when LIVE_MODE is enabled.

    CRITICAL SAFETY FUNCTION:
    - If LIVE_MODE is enabled but live_rollout_readiness is NOT GO:
      a) Logs CRITICAL error
      b) Activates kill switch (if activate_kill_switch_on_failure=True)
      c) Returns failure result

    This prevents accidental live trading when system is not ready.
    Paper trading / shadow collection are NOT affected.

    Args:
        guardrails: Optional guardrails instance (uses singleton if None)
        activate_kill_switch_on_failure: Whether to activate kill switch on failure

    Returns:
        StartupReadinessResult indicating if startup is safe for live trading
    """
    if guardrails is None:
        guardrails = get_live_guardrails()

    # If LIVE_MODE is not enabled, startup is always safe
    if not guardrails.config.live_mode:
        logger.info("Startup check: LIVE_MODE=false, skipping readiness validation")
        return StartupReadinessResult(
            passed=True,
            live_rollout_readiness="N/A",
            reasons=["LIVE_MODE is disabled - no readiness check required"],
            kill_switch_activated=False,
        )

    # LIVE_MODE is enabled - must validate readiness
    logger.warning("Startup check: LIVE_MODE=true, validating live rollout readiness...")

    try:
        from api.readiness_calculator import get_readiness

        result = get_readiness()
        live_rollout = result.live_rollout.readiness
        reasons = result.live_rollout.reasons

        if live_rollout == "GO":
            logger.info(
                f"Startup check PASSED: live_rollout_readiness=GO | "
                f"Reasons: {reasons}"
            )
            return StartupReadinessResult(
                passed=True,
                live_rollout_readiness=live_rollout,
                reasons=reasons,
                kill_switch_activated=False,
            )
        else:
            # NOT GO - this is a CRITICAL failure
            logger.critical(
                "=" * 60 + "\n"
                f"STARTUP SAFETY CHECK FAILED\n"
                f"live_rollout_readiness = {live_rollout} (required: GO)\n"
                f"Reasons:\n" + "\n".join(f"  - {r}" for r in reasons) + "\n"
                f"LIVE_MODE is enabled but system is NOT ready.\n"
                + "=" * 60
            )

            kill_switch_activated = False
            if activate_kill_switch_on_failure:
                guardrails.activate_kill_switch(
                    reason=f"Startup safety: live_rollout_readiness={live_rollout}"
                )
                kill_switch_activated = True
                logger.critical(
                    "KILL SWITCH ACTIVATED: Live trading blocked until readiness=GO"
                )

            return StartupReadinessResult(
                passed=False,
                live_rollout_readiness=live_rollout,
                reasons=reasons,
                kill_switch_activated=kill_switch_activated,
            )

    except ImportError as e:
        # Readiness calculator not available
        logger.critical(
            f"Startup check FAILED: Could not import readiness calculator: {e}\n"
            "LIVE_MODE is enabled but readiness cannot be verified."
        )

        if activate_kill_switch_on_failure:
            guardrails.activate_kill_switch(
                reason="Startup safety: readiness calculator unavailable"
            )

        return StartupReadinessResult(
            passed=False,
            live_rollout_readiness="UNKNOWN",
            reasons=[f"Readiness calculator unavailable: {e}"],
            kill_switch_activated=activate_kill_switch_on_failure,
        )

    except Exception as e:
        # Unexpected error
        logger.critical(
            f"Startup check FAILED: Unexpected error: {e}\n"
            "LIVE_MODE is enabled but readiness cannot be verified."
        )

        if activate_kill_switch_on_failure:
            guardrails.activate_kill_switch(
                reason=f"Startup safety: readiness check error: {e}"
            )

        return StartupReadinessResult(
            passed=False,
            live_rollout_readiness="ERROR",
            reasons=[f"Unexpected error: {e}"],
            kill_switch_activated=activate_kill_switch_on_failure,
        )


def validate_live_mode_startup() -> bool:
    """
    Convenience function to validate startup and return simple bool.

    Call this at the start of your main trading loop if LIVE_MODE might be enabled.

    Returns:
        True if safe to proceed with live trading, False otherwise
    """
    result = check_startup_readiness(activate_kill_switch_on_failure=True)
    return result.passed
