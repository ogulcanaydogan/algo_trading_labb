"""
Readiness Calculator for System Health Assessment.

Aggregates multiple health metrics into a unified readiness status:
- Shadow data collection health
- Live trading guardrails status
- Capital preservation state
- Turnover governor status
- Daily health reports (for live rollout)
- Execution realism / drift detection (for live rollout)

This module is READ-ONLY and provides observability only.

Includes both general readiness and April 1st live rollout readiness.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Readiness Status Constants
# =============================================================================

READINESS_GO = "GO"
READINESS_CONDITIONAL = "CONDITIONAL"
READINESS_NO_GO = "NO_GO"

# Capital preservation levels that trigger NO_GO
CRITICAL_PRESERVATION_LEVELS = {"LOCKDOWN", "CRISIS"}

# General readiness thresholds
MIN_STREAK_FOR_GO = 7

# Live rollout thresholds (stricter for April 1st)
LIVE_ROLLOUT_MIN_STREAK = 14
LIVE_ROLLOUT_MIN_WEEKS = 2
TURNOVER_BLOCK_RATE_MIN = 5.0   # Below this = too loose
TURNOVER_BLOCK_RATE_MAX = 70.0  # Above this = over-throttling


# =============================================================================
# Component Status Data Classes
# =============================================================================


@dataclass
class ShadowStatus:
    """Shadow data collection status."""

    paper_live_decisions_today: int = 0
    paper_live_decisions_7d: int = 0
    paper_live_days_streak: int = 0
    paper_live_weeks_counted: int = 0
    heartbeat_recent: int = 0
    overall_health: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_live_decisions_today": self.paper_live_decisions_today,
            "paper_live_decisions_7d": self.paper_live_decisions_7d,
            "paper_live_days_streak": self.paper_live_days_streak,
            "paper_live_weeks_counted": self.paper_live_weeks_counted,
            "heartbeat_recent": self.heartbeat_recent,
            "overall_health": self.overall_health,
        }


@dataclass
class LiveStatus:
    """Live trading guardrails status."""

    live_mode_enabled: bool = False
    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    daily_trades_remaining: int = 3
    overall_status: str = "SAFE"
    symbol_allowlist: List[str] = field(default_factory=lambda: ["ETH/USDT"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "live_mode_enabled": self.live_mode_enabled,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "daily_trades_remaining": self.daily_trades_remaining,
            "overall_status": self.overall_status,
            "symbol_allowlist": self.symbol_allowlist,
        }


@dataclass
class TurnoverStatus:
    """Turnover governor status."""

    enabled: bool = False
    symbols_configured: int = 0
    total_blocks_today: int = 0
    total_decisions_today: int = 0
    block_rate_pct: float = 0.0
    per_symbol_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "symbols_configured": self.symbols_configured,
            "total_blocks_today": self.total_blocks_today,
            "total_decisions_today": self.total_decisions_today,
            "block_rate_pct": round(self.block_rate_pct, 1),
            "per_symbol_config": self.per_symbol_config,
        }


@dataclass
class CapitalPreservationStatus:
    """Capital preservation mode status."""

    current_level: str = "NORMAL"
    last_escalation: Optional[str] = None
    restrictions_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_level": self.current_level,
            "last_escalation": self.last_escalation,
            "restrictions_active": self.restrictions_active,
        }


@dataclass
class DailyReportsStatus:
    """Daily health reports status for live rollout checks."""

    latest_report_timestamp: Optional[str] = None
    latest_report_age_hours: float = 999.0
    reports_last_24h: bool = False
    critical_alerts_14d: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latest_report_timestamp": self.latest_report_timestamp,
            "latest_report_age_hours": round(self.latest_report_age_hours, 1),
            "reports_last_24h": self.reports_last_24h,
            "critical_alerts_14d": self.critical_alerts_14d,
        }


@dataclass
class ExecutionRealismStatus:
    """Execution realism / drift detection status."""

    drift_detected: bool = False
    slippage_7d_avg: float = 0.0
    slippage_prior_7d_avg: float = 0.0
    slippage_worsened: bool = False
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "slippage_7d_avg": round(self.slippage_7d_avg, 4),
            "slippage_prior_7d_avg": round(self.slippage_prior_7d_avg, 4),
            "slippage_worsened": self.slippage_worsened,
            "available": self.available,
        }


@dataclass
class LiveRolloutResult:
    """Result of live rollout readiness calculation."""

    readiness: Literal["GO", "CONDITIONAL", "NO_GO"]
    reasons: List[str]
    next_actions: List[str]


@dataclass
class ReadinessResult:
    """Result of readiness calculation."""

    overall_readiness: Literal["GO", "CONDITIONAL", "NO_GO"]
    reasons: List[str]
    recommended_next_actions: List[str]
    live_rollout: LiveRolloutResult
    shadow: ShadowStatus
    live: LiveStatus
    turnover: TurnoverStatus
    capital_preservation: CapitalPreservationStatus
    daily_reports: DailyReportsStatus
    execution_realism: ExecutionRealismStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_readiness": self.overall_readiness,
            "reasons": self.reasons,
            "recommended_next_actions": self.recommended_next_actions,
            "live_rollout_readiness": self.live_rollout.readiness,
            "live_rollout_reasons": self.live_rollout.reasons,
            "live_rollout_next_actions": self.live_rollout.next_actions,
            "components": {
                "shadow": self.shadow.to_dict(),
                "live": self.live.to_dict(),
                "turnover": self.turnover.to_dict(),
                "capital_preservation": self.capital_preservation.to_dict(),
                "daily_reports": self.daily_reports.to_dict(),
                "execution_realism": self.execution_realism.to_dict(),
            },
        }


# =============================================================================
# Data Fetchers
# =============================================================================


def get_shadow_status() -> ShadowStatus:
    """Fetch shadow data collection status."""
    try:
        from .shadow_health_metrics import get_shadow_health_metrics

        metrics = get_shadow_health_metrics()
        return ShadowStatus(
            paper_live_decisions_today=metrics.paper_live_decisions_today,
            paper_live_decisions_7d=metrics.paper_live_decisions_7d,
            paper_live_days_streak=metrics.paper_live_days_streak,
            paper_live_weeks_counted=metrics.paper_live_weeks_counted,
            heartbeat_recent=metrics.heartbeat_recent,
            overall_health=metrics.overall_health,
        )
    except Exception as e:
        logger.warning(f"Failed to get shadow status: {e}")
        return ShadowStatus(overall_health="UNAVAILABLE")


def get_live_status() -> LiveStatus:
    """Fetch live trading guardrails status."""
    try:
        from bot.live_trading_guardrails import get_live_guardrails

        guardrails = get_live_guardrails()
        status = guardrails.get_status()

        return LiveStatus(
            live_mode_enabled=status.get("live_mode_enabled", False),
            kill_switch_active=status.get("kill_switch_active", False),
            kill_switch_reason=status.get("kill_switch_reason"),
            daily_trades_remaining=status.get("daily_trades", {}).get("remaining", 3),
            overall_status=status.get("overall_status", "SAFE")
            if not status.get("kill_switch_active")
            else "BLOCKED",
            symbol_allowlist=status.get("symbol_allowlist", ["ETH/USDT"]),
        )
    except ImportError:
        logger.debug("Live guardrails not available")
        return LiveStatus()
    except Exception as e:
        logger.warning(f"Failed to get live status: {e}")
        return LiveStatus()


def get_turnover_status() -> TurnoverStatus:
    """Fetch turnover governor status with block rate calculation."""
    try:
        from bot.turnover_governor import get_turnover_governor

        governor = get_turnover_governor()
        if not governor:
            return TurnoverStatus()

        # Get per-symbol configs and aggregate stats
        per_symbol = {}
        total_blocks = 0
        total_decisions = 0

        try:
            all_configs = governor.get_all_symbol_configs()
            for symbol, config in all_configs.items():
                status = governor.get_symbol_status(symbol)
                decisions = status.get("decisions_today", 0)
                blocked = status.get("blocked_today", 0)
                per_symbol[symbol] = {
                    "effective_interval": config.min_interval_minutes,
                    "effective_max_decisions": config.max_decisions_per_day,
                    "effective_ev_multiple": config.min_ev_cost_multiple,
                    "decisions_today": decisions,
                    "blocked_today": blocked,
                }
                total_decisions += decisions
                total_blocks += blocked
        except Exception:
            pass

        # Calculate block rate
        total_attempted = total_decisions + total_blocks
        block_rate = (total_blocks / total_attempted * 100) if total_attempted > 0 else 0.0

        return TurnoverStatus(
            enabled=governor.config.enabled,
            symbols_configured=len(per_symbol),
            total_blocks_today=total_blocks,
            total_decisions_today=total_decisions,
            block_rate_pct=block_rate,
            per_symbol_config=per_symbol,
        )
    except ImportError:
        logger.debug("Turnover governor not available")
        return TurnoverStatus()
    except Exception as e:
        logger.warning(f"Failed to get turnover status: {e}")
        return TurnoverStatus()


def get_capital_preservation_status() -> CapitalPreservationStatus:
    """Fetch capital preservation mode status."""
    try:
        from bot.safety.capital_preservation import get_capital_preservation

        preservation = get_capital_preservation()
        if not preservation:
            return CapitalPreservationStatus()

        # Access level via _state.level (internal state)
        level = "NORMAL"
        last_escalation = None

        if hasattr(preservation, '_state') and preservation._state:
            state = preservation._state
            if hasattr(state, 'level'):
                level = state.level.value if hasattr(state.level, 'value') else str(state.level)
            if hasattr(state, 'last_escalation_time'):
                last_escalation = state.last_escalation_time

        return CapitalPreservationStatus(
            current_level=level.upper(),
            last_escalation=last_escalation,
            restrictions_active=level.upper() not in ("NORMAL",),
        )
    except ImportError:
        logger.debug("Capital preservation not available")
        return CapitalPreservationStatus()
    except Exception as e:
        logger.warning(f"Failed to get capital preservation status: {e}")
        return CapitalPreservationStatus()


def get_daily_reports_status() -> DailyReportsStatus:
    """Fetch daily health reports status."""
    try:
        reports_dir = Path(os.getenv("SHADOW_REPORTS_DIR", "data/shadow_reports"))
        now = datetime.now(timezone.utc)

        # Find latest report
        latest_report = None
        latest_timestamp = None
        critical_count = 0

        if reports_dir.exists():
            report_files = sorted(reports_dir.glob("shadow_health_*.json"), reverse=True)

            for report_file in report_files[:14]:  # Check last 14 reports
                try:
                    import json
                    with open(report_file) as f:
                        data = json.load(f)

                    # Parse timestamp from filename or content
                    ts_str = data.get("timestamp") or data.get("report_timestamp")
                    if ts_str:
                        from dateutil.parser import parse
                        ts = parse(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)

                        if latest_timestamp is None or ts > latest_timestamp:
                            latest_timestamp = ts
                            latest_report = report_file

                        # Count critical alerts in last 14 days
                        if (now - ts).days <= 14:
                            health = data.get("overall_health", "")
                            if health == "CRITICAL":
                                critical_count += 1
                except Exception:
                    continue

        # Calculate age
        age_hours = 999.0
        reports_last_24h = False
        if latest_timestamp:
            age_hours = (now - latest_timestamp).total_seconds() / 3600
            reports_last_24h = age_hours <= 24.0

        return DailyReportsStatus(
            latest_report_timestamp=latest_timestamp.isoformat() if latest_timestamp else None,
            latest_report_age_hours=age_hours,
            reports_last_24h=reports_last_24h,
            critical_alerts_14d=critical_count,
        )
    except Exception as e:
        logger.warning(f"Failed to get daily reports status: {e}")
        return DailyReportsStatus()


def get_execution_realism_status() -> ExecutionRealismStatus:
    """Fetch execution realism / drift detection status."""
    try:
        # Try to get slippage metrics from execution simulator or forensics
        from bot.execution.execution_simulator import get_execution_simulator

        simulator = get_execution_simulator()
        if not simulator:
            return ExecutionRealismStatus(available=False)

        # Get slippage data if available
        slippage_7d = 0.0
        slippage_prior_7d = 0.0

        try:
            if hasattr(simulator, 'get_slippage_stats'):
                stats = simulator.get_slippage_stats()
                slippage_7d = stats.get("avg_slippage_7d", 0.0)
                slippage_prior_7d = stats.get("avg_slippage_prior_7d", 0.0)
        except Exception:
            pass

        # Detect drift: slippage worsened by >50%
        slippage_worsened = False
        if slippage_prior_7d > 0:
            change_pct = (slippage_7d - slippage_prior_7d) / slippage_prior_7d
            slippage_worsened = change_pct > 0.5  # 50% worse

        return ExecutionRealismStatus(
            drift_detected=slippage_worsened,
            slippage_7d_avg=slippage_7d,
            slippage_prior_7d_avg=slippage_prior_7d,
            slippage_worsened=slippage_worsened,
            available=True,
        )
    except ImportError:
        logger.debug("Execution simulator not available")
        return ExecutionRealismStatus(available=False)
    except Exception as e:
        logger.warning(f"Failed to get execution realism status: {e}")
        return ExecutionRealismStatus(available=False)


# =============================================================================
# Live Rollout Readiness Calculator (April 1st oriented)
# =============================================================================


def calculate_live_rollout_readiness(
    shadow: ShadowStatus,
    live: LiveStatus,
    turnover: TurnoverStatus,
    capital_preservation: CapitalPreservationStatus,
    daily_reports: DailyReportsStatus,
    execution_realism: ExecutionRealismStatus,
) -> LiveRolloutResult:
    """
    Calculate live rollout readiness with stricter requirements.

    NO_GO if:
    - kill_switch_active
    - capital_preservation in LOCKDOWN/CRISIS
    - shadow overall_health == CRITICAL
    - live_mode_enabled == true AND symbol_allowlist empty (misconfig)
    - daily health reports missing in last 24h

    CONDITIONAL if:
    - paper_live_days_streak < 14
    - paper_live_weeks_counted < 2
    - heartbeat_recent == 0
    - turnover blocks today > 70% of decisions (over-throttling) OR < 5% (too loose)
    - execution realism degradation (drift flag)

    GO if:
    - streak >= 14 days, weeks_counted >= 2
    - heartbeat_recent == 1
    - live endpoint SAFE, kill switch not active
    - turnover block-rate in healthy band (5–70%)
    - no CRITICAL alerts in last 14 days
    """
    reasons: List[str] = []
    next_actions: List[str] = []
    readiness: Literal["GO", "CONDITIONAL", "NO_GO"] = READINESS_GO

    # === NO_GO Checks (Critical) ===

    # 1. Kill switch active
    if live.kill_switch_active:
        readiness = READINESS_NO_GO
        reason = "Kill switch is active"
        if live.kill_switch_reason:
            reason += f": {live.kill_switch_reason}"
        reasons.append(reason)
        next_actions.append("Investigate kill switch activation and deactivate when safe")

    # 2. Capital preservation in critical state
    if capital_preservation.current_level.upper() in CRITICAL_PRESERVATION_LEVELS:
        readiness = READINESS_NO_GO
        reasons.append(f"Capital preservation in {capital_preservation.current_level} mode")
        next_actions.append("Wait for capital preservation to return to NORMAL")

    # 3. Shadow health critical
    if shadow.overall_health == "CRITICAL":
        readiness = READINESS_NO_GO
        reasons.append("Shadow data collection health is CRITICAL")
        next_actions.append("Investigate shadow collector issues immediately")

    # 4. Live mode misconfiguration (enabled but empty allowlist)
    if live.live_mode_enabled and not live.symbol_allowlist:
        readiness = READINESS_NO_GO
        reasons.append("Live mode enabled but symbol allowlist is empty (misconfiguration)")
        next_actions.append("Configure LIVE_SYMBOL_ALLOWLIST or disable LIVE_MODE")

    # 5. Daily health reports missing
    if not daily_reports.reports_last_24h:
        readiness = READINESS_NO_GO
        age_str = f"{daily_reports.latest_report_age_hours:.0f}h" if daily_reports.latest_report_age_hours < 999 else "never"
        reasons.append(f"Daily health reports missing in last 24h (last report: {age_str} ago)")
        next_actions.append("Run daily shadow health report or verify cron job")

    # If NO_GO, return early
    if readiness == READINESS_NO_GO:
        return LiveRolloutResult(
            readiness=readiness,
            reasons=reasons,
            next_actions=next_actions,
        )

    # === CONDITIONAL Checks ===

    # 6. Insufficient streak (< 14 days for live rollout)
    if shadow.paper_live_days_streak < LIVE_ROLLOUT_MIN_STREAK:
        readiness = READINESS_CONDITIONAL
        reasons.append(
            f"PAPER_LIVE streak ({shadow.paper_live_days_streak} days) "
            f"below minimum ({LIVE_ROLLOUT_MIN_STREAK} days)"
        )
        days_needed = LIVE_ROLLOUT_MIN_STREAK - shadow.paper_live_days_streak
        next_actions.append(f"Continue PAPER_LIVE trading for {days_needed} more consecutive days")

    # 7. Insufficient weeks (< 2 weeks for live rollout)
    if shadow.paper_live_weeks_counted < LIVE_ROLLOUT_MIN_WEEKS:
        if readiness != READINESS_NO_GO:
            readiness = READINESS_CONDITIONAL
        reasons.append(
            f"PAPER_LIVE weeks counted ({shadow.paper_live_weeks_counted}) "
            f"below minimum ({LIVE_ROLLOUT_MIN_WEEKS})"
        )
        weeks_needed = LIVE_ROLLOUT_MIN_WEEKS - shadow.paper_live_weeks_counted
        next_actions.append(f"Accumulate {weeks_needed} more week(s) of PAPER_LIVE data")

    # 8. Heartbeat not recent
    if shadow.heartbeat_recent == 0:
        if readiness != READINESS_NO_GO:
            readiness = READINESS_CONDITIONAL
        reasons.append("Shadow collector heartbeat not recent (>2 hours)")
        next_actions.append("Verify shadow collector is running")

    # 9. Turnover block rate out of healthy band
    if turnover.enabled and (turnover.total_decisions_today + turnover.total_blocks_today) > 0:
        if turnover.block_rate_pct > TURNOVER_BLOCK_RATE_MAX:
            if readiness != READINESS_NO_GO:
                readiness = READINESS_CONDITIONAL
            reasons.append(
                f"Turnover block rate ({turnover.block_rate_pct:.1f}%) exceeds "
                f"{TURNOVER_BLOCK_RATE_MAX}% (over-throttling)"
            )
            next_actions.append("Review turnover governor settings - may be too strict")
        elif turnover.block_rate_pct < TURNOVER_BLOCK_RATE_MIN:
            if readiness != READINESS_NO_GO:
                readiness = READINESS_CONDITIONAL
            reasons.append(
                f"Turnover block rate ({turnover.block_rate_pct:.1f}%) below "
                f"{TURNOVER_BLOCK_RATE_MIN}% (too loose)"
            )
            next_actions.append("Review turnover governor settings - may need tightening")

    # 10. Execution realism drift
    if execution_realism.available and execution_realism.drift_detected:
        if readiness != READINESS_NO_GO:
            readiness = READINESS_CONDITIONAL
        reasons.append(
            f"Execution realism degradation detected "
            f"(slippage {execution_realism.slippage_7d_avg:.4f} vs prior {execution_realism.slippage_prior_7d_avg:.4f})"
        )
        next_actions.append("Investigate execution model drift before live rollout")

    # 11. Critical alerts in last 14 days
    if daily_reports.critical_alerts_14d > 0:
        if readiness != READINESS_NO_GO:
            readiness = READINESS_CONDITIONAL
        reasons.append(f"{daily_reports.critical_alerts_14d} CRITICAL alert(s) in last 14 days")
        next_actions.append("Review and resolve recent CRITICAL alerts before live rollout")

    # === GO Status ===

    if readiness == READINESS_GO:
        reasons.append("All live rollout criteria met")
        next_actions.append("System ready for April 1st live rollout")

    return LiveRolloutResult(
        readiness=readiness,
        reasons=reasons,
        next_actions=next_actions,
    )


# =============================================================================
# Main Readiness Calculator
# =============================================================================


def calculate_readiness(
    shadow: Optional[ShadowStatus] = None,
    live: Optional[LiveStatus] = None,
    turnover: Optional[TurnoverStatus] = None,
    capital_preservation: Optional[CapitalPreservationStatus] = None,
    daily_reports: Optional[DailyReportsStatus] = None,
    execution_realism: Optional[ExecutionRealismStatus] = None,
) -> ReadinessResult:
    """
    Calculate system readiness based on all component statuses.

    General Readiness Logic (simple + strict):
    1. If kill_switch_active => NO_GO
    2. If capital preservation in LOCKDOWN/CRISIS => NO_GO
    3. If shadow overall_health == CRITICAL => NO_GO
    4. Else if heartbeat_recent == 0 => CONDITIONAL (shadow not running)
    5. Else if paper_live_days_streak < 7 => CONDITIONAL
    6. Else => GO

    Also calculates live_rollout_readiness with stricter April 1st requirements.

    Args:
        shadow: Shadow data collection status (fetched if None)
        live: Live trading guardrails status (fetched if None)
        turnover: Turnover governor status (fetched if None)
        capital_preservation: Capital preservation status (fetched if None)
        daily_reports: Daily reports status (fetched if None)
        execution_realism: Execution realism status (fetched if None)

    Returns:
        ReadinessResult with overall status, live rollout status, and recommendations
    """
    # Fetch statuses if not provided
    shadow = shadow or get_shadow_status()
    live = live or get_live_status()
    turnover = turnover or get_turnover_status()
    capital_preservation = capital_preservation or get_capital_preservation_status()
    daily_reports = daily_reports or get_daily_reports_status()
    execution_realism = execution_realism or get_execution_realism_status()

    reasons: List[str] = []
    recommended_actions: List[str] = []
    overall_readiness: Literal["GO", "CONDITIONAL", "NO_GO"] = READINESS_GO

    # === NO_GO Checks (Critical) ===

    # Check 1: Kill switch active
    if live.kill_switch_active:
        overall_readiness = READINESS_NO_GO
        reason = "Kill switch is active"
        if live.kill_switch_reason:
            reason += f": {live.kill_switch_reason}"
        reasons.append(reason)
        recommended_actions.append("Investigate kill switch activation and deactivate when safe")

    # Check 2: Capital preservation in critical state
    if capital_preservation.current_level.upper() in CRITICAL_PRESERVATION_LEVELS:
        overall_readiness = READINESS_NO_GO
        reasons.append(f"Capital preservation in {capital_preservation.current_level} mode")
        recommended_actions.append("Wait for capital preservation to return to NORMAL")
        if capital_preservation.last_escalation:
            recommended_actions.append(f"Review escalation from {capital_preservation.last_escalation}")

    # Check 3: Shadow health critical
    if shadow.overall_health == "CRITICAL":
        overall_readiness = READINESS_NO_GO
        reasons.append("Shadow data collection health is CRITICAL")
        recommended_actions.append("Investigate shadow collector issues immediately")

    # If NO_GO, calculate live rollout and return
    if overall_readiness == READINESS_NO_GO:
        live_rollout = calculate_live_rollout_readiness(
            shadow, live, turnover, capital_preservation, daily_reports, execution_realism
        )
        return ReadinessResult(
            overall_readiness=overall_readiness,
            reasons=reasons,
            recommended_next_actions=recommended_actions,
            live_rollout=live_rollout,
            shadow=shadow,
            live=live,
            turnover=turnover,
            capital_preservation=capital_preservation,
            daily_reports=daily_reports,
            execution_realism=execution_realism,
        )

    # === CONDITIONAL Checks ===

    # Check 4: Heartbeat not recent (shadow not running)
    if shadow.heartbeat_recent == 0:
        overall_readiness = READINESS_CONDITIONAL
        reasons.append("Shadow collector heartbeat not recent (>2 hours)")
        recommended_actions.append("Verify shadow collector is running")

    # Check 5: Insufficient streak
    if shadow.paper_live_days_streak < MIN_STREAK_FOR_GO:
        if overall_readiness != READINESS_NO_GO:
            overall_readiness = READINESS_CONDITIONAL
        reasons.append(
            f"PAPER_LIVE streak ({shadow.paper_live_days_streak} days) "
            f"below minimum ({MIN_STREAK_FOR_GO} days)"
        )
        days_needed = MIN_STREAK_FOR_GO - shadow.paper_live_days_streak
        recommended_actions.append(f"Continue PAPER_LIVE trading for {days_needed} more consecutive days")

    # === Additional Informational Checks ===

    # Shadow health warning (not a blocker but noted)
    if shadow.overall_health == "WARNING" and "Shadow" not in " ".join(reasons):
        reasons.append("Shadow data collection health is WARNING")

    # Live mode status
    if live.live_mode_enabled:
        reasons.append("Live trading mode is ENABLED")
        if live.daily_trades_remaining == 0:
            recommended_actions.append("Daily live trade limit reached - trades will resume tomorrow")

    # === GO Status ===

    if overall_readiness == READINESS_GO:
        if not reasons:
            reasons.append("All systems operational")
        recommended_actions.append("System ready for live trading transition when appropriate")

    # Calculate live rollout readiness
    live_rollout = calculate_live_rollout_readiness(
        shadow, live, turnover, capital_preservation, daily_reports, execution_realism
    )

    return ReadinessResult(
        overall_readiness=overall_readiness,
        reasons=reasons,
        recommended_next_actions=recommended_actions,
        live_rollout=live_rollout,
        shadow=shadow,
        live=live,
        turnover=turnover,
        capital_preservation=capital_preservation,
        daily_reports=daily_reports,
        execution_realism=execution_realism,
    )


def get_readiness() -> ReadinessResult:
    """
    Get current system readiness status.

    This is the main entry point for the readiness check.
    Fetches all component statuses and calculates readiness.

    Returns:
        ReadinessResult with overall status, live rollout status, and component details
    """
    return calculate_readiness()
