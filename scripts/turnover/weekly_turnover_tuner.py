#!/usr/bin/env python3
"""
Weekly Turnover Governor Tuner.

SAFE, bounded auto-tuning of turnover governor parameters based on
last 7 days of PAPER_LIVE data.

Behavior:
- Analyzes block rates, cost drag, and decision patterns
- Recommends config changes within strict per-symbol bounds
- Only applies changes if safety conditions are met
- Writes audit artifact and rollback file

Safety Conditions:
- PAPER_LIVE weeks_counted >= 1
- No CRITICAL health alerts in the last 7 days

CRITICAL: This is tuning only. No execution authority.
All changes are recommendations that must pass safety gates.
"""

import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# TUNING BOUNDS - STRICT, NON-NEGOTIABLE
# =============================================================================

@dataclass
class SymbolTuningBounds:
    """Strict bounds for per-symbol tuning parameters."""

    min_interval_min: float  # Minimum allowed min_interval_minutes
    min_interval_max: float  # Maximum allowed min_interval_minutes
    max_decisions_min: int   # Minimum allowed max_decisions_per_day
    max_decisions_max: int   # Maximum allowed max_decisions_per_day
    ev_multiple_min: float   # Minimum allowed min_ev_cost_multiple
    ev_multiple_max: float   # Maximum allowed min_ev_cost_multiple

    def clamp_interval(self, value: float) -> float:
        """Clamp interval to bounds."""
        return max(self.min_interval_min, min(self.min_interval_max, value))

    def clamp_max_decisions(self, value: int) -> int:
        """Clamp max_decisions to bounds."""
        return max(self.max_decisions_min, min(self.max_decisions_max, value))

    def clamp_ev_multiple(self, value: float) -> float:
        """Clamp ev_multiple to bounds."""
        return max(self.ev_multiple_min, min(self.ev_multiple_max, value))


# Per-symbol bounds (STRICT)
BTC_BOUNDS = SymbolTuningBounds(
    min_interval_min=30.0,    # At least 30 minutes between decisions
    min_interval_max=240.0,   # At most 4 hours between decisions
    max_decisions_min=2,      # At least 2 decisions per day
    max_decisions_max=6,      # At most 6 decisions per day
    ev_multiple_min=2.5,      # At least 2.5x EV/cost ratio
    ev_multiple_max=4.0,      # At most 4.0x EV/cost ratio
)

ETH_BOUNDS = SymbolTuningBounds(
    min_interval_min=10.0,    # At least 10 minutes between decisions
    min_interval_max=60.0,    # At most 1 hour between decisions
    max_decisions_min=6,      # At least 6 decisions per day
    max_decisions_max=12,     # At most 12 decisions per day
    ev_multiple_min=1.8,      # At least 1.8x EV/cost ratio
    ev_multiple_max=3.0,      # At most 3.0x EV/cost ratio
)

# Default bounds for other symbols (conservative)
DEFAULT_BOUNDS = SymbolTuningBounds(
    min_interval_min=15.0,
    min_interval_max=120.0,
    max_decisions_min=4,
    max_decisions_max=10,
    ev_multiple_min=2.0,
    ev_multiple_max=3.5,
)


def get_bounds_for_symbol(symbol: str) -> SymbolTuningBounds:
    """Get tuning bounds for a symbol."""
    symbol_upper = symbol.upper()
    if "BTC" in symbol_upper:
        return BTC_BOUNDS
    elif "ETH" in symbol_upper:
        return ETH_BOUNDS
    else:
        return DEFAULT_BOUNDS


# =============================================================================
# TARGET MINIMUM DECISIONS PER DAY - For safe EV/cost tuning
# =============================================================================

# SAFETY RULE: High EV/cost block rate does NOT automatically reduce EV multiple.
# Instead, we only consider loosening OTHER parameters (interval/max_decisions)
# if decisions_per_day is BELOW these targets. This preserves trade quality.

TARGET_MIN_DECISIONS_PER_DAY_DEFAULT = 3  # Most symbols
TARGET_MIN_DECISIONS_PER_DAY_BTC = 1      # BTC is more selective
TARGET_MIN_DECISIONS_PER_DAY_ETH = 3      # ETH similar to default


def get_target_min_decisions_for_symbol(symbol: str) -> int:
    """Get target minimum decisions per day for a symbol."""
    symbol_upper = symbol.upper()
    if "BTC" in symbol_upper:
        return TARGET_MIN_DECISIONS_PER_DAY_BTC
    elif "ETH" in symbol_upper:
        return TARGET_MIN_DECISIONS_PER_DAY_ETH
    else:
        return TARGET_MIN_DECISIONS_PER_DAY_DEFAULT


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SymbolTuningData:
    """Aggregated tuning data for a symbol."""

    symbol: str
    days_with_data: int = 0
    total_decisions: int = 0
    total_blocked: int = 0
    blocked_by_interval: int = 0
    blocked_by_daily_limit: int = 0
    blocked_by_ev_cost: int = 0
    estimated_cost_drag_avoided: float = 0.0

    # Current config
    current_min_interval: float = 15.0
    current_max_decisions: int = 10
    current_ev_multiple: float = 2.0

    # Performance metrics (if available)
    avg_daily_pnl: Optional[float] = None
    win_rate: Optional[float] = None

    @property
    def decisions_per_day(self) -> float:
        """Average decisions per day (used for safe EV/cost tuning)."""
        if self.days_with_data == 0:
            return 0.0
        return self.total_decisions / self.days_with_data

    @property
    def block_rate(self) -> float:
        """Total block rate."""
        total = self.total_decisions + self.total_blocked
        return self.total_blocked / total if total > 0 else 0.0

    @property
    def interval_block_rate(self) -> float:
        """Block rate due to interval."""
        total = self.total_decisions + self.total_blocked
        return self.blocked_by_interval / total if total > 0 else 0.0

    @property
    def daily_limit_block_rate(self) -> float:
        """Block rate due to daily limit."""
        total = self.total_decisions + self.total_blocked
        return self.blocked_by_daily_limit / total if total > 0 else 0.0

    @property
    def ev_cost_block_rate(self) -> float:
        """Block rate due to EV/cost ratio."""
        total = self.total_decisions + self.total_blocked
        return self.blocked_by_ev_cost / total if total > 0 else 0.0


@dataclass
class TuningRationale:
    """Rationale for a tuning decision, especially for EV/cost safety rules."""

    rule_applied: str  # e.g., "ev_cost_safety", "interval_loosening", "tightening"
    decisions_per_day: float
    target_min_decisions: int
    ev_cost_block_rate: float
    action_taken: str  # e.g., "no_change", "loosen_interval", "increase_ev"
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_applied": self.rule_applied,
            "decisions_per_day": round(self.decisions_per_day, 2),
            "target_min_decisions": self.target_min_decisions,
            "ev_cost_block_rate": round(self.ev_cost_block_rate, 4),
            "action_taken": self.action_taken,
            "explanation": self.explanation,
        }


@dataclass
class TuningRecommendation:
    """Recommended config change for a symbol."""

    symbol: str
    parameter: str  # "min_interval", "max_decisions", or "ev_multiple"
    current_value: float
    recommended_value: float
    reason: str
    confidence: float = 0.5  # 0-1, how confident we are in this recommendation
    tuning_rationale: Optional[TuningRationale] = None  # For EV/cost safety rules

    @property
    def change_pct(self) -> float:
        """Percentage change from current to recommended."""
        if self.current_value == 0:
            return 0.0
        return (self.recommended_value - self.current_value) / self.current_value * 100

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "change_pct": round(self.change_pct, 2),
            "reason": self.reason,
            "confidence": self.confidence,
        }
        if self.tuning_rationale:
            result["tuning_rationale"] = self.tuning_rationale.to_dict()
        return result


@dataclass
class TuningResult:
    """Result of weekly tuning analysis."""

    timestamp: str
    week_start: str
    week_end: str

    # Safety gate results
    paper_live_weeks_counted: int = 0
    safety_gate_passed: bool = False
    safety_gate_reason: str = ""
    critical_alerts_found: List[str] = field(default_factory=list)

    # Data summary
    symbols_analyzed: List[str] = field(default_factory=list)
    total_decisions_week: int = 0
    total_blocked_week: int = 0
    total_cost_drag_avoided: float = 0.0

    # Recommendations
    recommendations: List[TuningRecommendation] = field(default_factory=list)
    recommendations_applied: bool = False

    # Rollback info
    rollback_file: Optional[str] = None
    previous_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "week_start": self.week_start,
            "week_end": self.week_end,
            "safety_gate": {
                "passed": self.safety_gate_passed,
                "reason": self.safety_gate_reason,
                "paper_live_weeks_counted": self.paper_live_weeks_counted,
                "critical_alerts_found": self.critical_alerts_found,
            },
            "data_summary": {
                "symbols_analyzed": self.symbols_analyzed,
                "total_decisions_week": self.total_decisions_week,
                "total_blocked_week": self.total_blocked_week,
                "total_cost_drag_avoided": round(self.total_cost_drag_avoided, 2),
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
            "recommendations_applied": self.recommendations_applied,
            "rollback": {
                "file": self.rollback_file,
                "previous_config": self.previous_config,
            },
        }


# =============================================================================
# WEEKLY TURNOVER TUNER
# =============================================================================

class WeeklyTurnoverTuner:
    """
    Weekly tuner for turnover governor parameters.

    Analyzes last 7 days of data and produces bounded recommendations.
    """

    def __init__(
        self,
        reports_dir: Path = Path("data/reports"),
        turnover_state_path: Path = Path("data/turnover_governor_state.json"),
        output_dir: Path = Path("data/reports"),
    ):
        self.reports_dir = reports_dir
        self.turnover_state_path = turnover_state_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tuning thresholds
        self.HIGH_BLOCK_RATE = 0.5  # If >50% blocked, consider loosening
        self.LOW_BLOCK_RATE = 0.1   # If <10% blocked, consider tightening
        self.MIN_DATA_POINTS = 20   # Need at least 20 decisions for analysis

    def run_tuning(self, apply_changes: bool = False) -> TuningResult:
        """
        Run the weekly tuning analysis.

        Args:
            apply_changes: If True, apply recommended changes (with rollback)

        Returns:
            TuningResult with recommendations and status
        """
        today = datetime.now()
        week_end = today.strftime("%Y-%m-%d")
        week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")

        result = TuningResult(
            timestamp=today.isoformat(),
            week_start=week_start,
            week_end=week_end,
        )

        # Step 1: Check safety gates
        logger.info("=" * 60)
        logger.info("Weekly Turnover Governor Tuner")
        logger.info("=" * 60)
        logger.info(f"Analyzing period: {week_start} to {week_end}")

        if not self._check_safety_gates(result):
            logger.warning(f"Safety gate FAILED: {result.safety_gate_reason}")
            self._save_result(result)
            return result

        logger.info("Safety gates PASSED")

        # Step 2: Load and aggregate data
        symbol_data = self._load_weekly_data(result)
        if not symbol_data:
            result.safety_gate_passed = False
            result.safety_gate_reason = "No symbol data found for analysis"
            logger.warning(result.safety_gate_reason)
            self._save_result(result)
            return result

        # Step 3: Generate recommendations
        for symbol, data in symbol_data.items():
            self._analyze_symbol(data, result)

        logger.info(f"Generated {len(result.recommendations)} recommendations")

        # Step 4: Optionally apply changes
        if apply_changes and result.recommendations:
            self._apply_changes(result)

        # Step 5: Save result
        self._save_result(result)

        return result

    def _check_safety_gates(self, result: TuningResult) -> bool:
        """
        Check all safety gates before tuning.

        Returns True if all gates pass.
        """
        # Gate 1: Check PAPER_LIVE weeks_counted
        try:
            from api.shadow_health_metrics import ShadowHealthMetricsCalculator
            calculator = ShadowHealthMetricsCalculator(reports_dir=self.reports_dir)
            metrics = calculator.calculate_metrics()
            result.paper_live_weeks_counted = metrics.paper_live_weeks_counted

            if metrics.paper_live_weeks_counted < 1:
                result.safety_gate_reason = (
                    f"PAPER_LIVE weeks_counted={metrics.paper_live_weeks_counted} < 1. "
                    "Need at least 1 week of PAPER_LIVE data before tuning."
                )
                return False

        except Exception as e:
            logger.warning(f"Could not check PAPER_LIVE metrics: {e}")
            # If we can't check, fail safe
            result.safety_gate_reason = f"Could not verify PAPER_LIVE metrics: {e}"
            return False

        # Gate 2: Check for CRITICAL alerts in last 7 days
        critical_alerts = self._find_critical_alerts_last_7_days()
        if critical_alerts:
            result.critical_alerts_found = critical_alerts
            result.safety_gate_reason = (
                f"Found {len(critical_alerts)} CRITICAL alerts in last 7 days. "
                "Tuning disabled until alerts are resolved."
            )
            return False

        result.safety_gate_passed = True
        result.safety_gate_reason = "All safety gates passed"
        return True

    def _find_critical_alerts_last_7_days(self) -> List[str]:
        """Find any CRITICAL alerts in the last 7 days of reports."""
        critical_alerts = []
        today = datetime.now()

        for days_ago in range(7):
            date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            report_path = self.reports_dir / f"daily_shadow_health_{date}.json"

            if report_path.exists():
                try:
                    with open(report_path) as f:
                        report = json.load(f)

                    # Check overall health
                    if report.get("summary", {}).get("overall_health") == "CRITICAL":
                        critical_alerts.append(f"{date}: CRITICAL health status")

                    # Check individual alerts
                    for alert in report.get("summary", {}).get("alerts", []):
                        if alert.get("severity") == "HIGH":
                            critical_alerts.append(f"{date}: {alert.get('message', 'Unknown')}")

                except Exception as e:
                    logger.warning(f"Could not read report {report_path}: {e}")

        return critical_alerts

    def _load_weekly_data(self, result: TuningResult) -> Dict[str, SymbolTuningData]:
        """Load and aggregate data from last 7 days."""
        symbol_data: Dict[str, SymbolTuningData] = {}
        today = datetime.now()

        # Load daily reports
        for days_ago in range(7):
            date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            report_path = self.reports_dir / f"daily_shadow_health_{date}.json"

            if not report_path.exists():
                continue

            try:
                with open(report_path) as f:
                    report = json.load(f)

                # Extract turnover data
                tg = report.get("turnover_governor", {})
                per_symbol = tg.get("per_symbol_blocked", {})
                per_symbol_configs = tg.get("per_symbol_configs", {})

                for symbol, stats in per_symbol.items():
                    if symbol.startswith("_"):  # Skip metadata keys
                        continue

                    if symbol not in symbol_data:
                        symbol_data[symbol] = SymbolTuningData(symbol=symbol)

                        # Get current config
                        cfg = per_symbol_configs.get(symbol, {})
                        symbol_data[symbol].current_min_interval = cfg.get(
                            "min_interval_minutes", 15.0
                        )
                        symbol_data[symbol].current_max_decisions = cfg.get(
                            "max_decisions_per_day", 10
                        )
                        symbol_data[symbol].current_ev_multiple = cfg.get(
                            "min_ev_cost_multiple", 2.0
                        )

                    data = symbol_data[symbol]
                    data.days_with_data += 1
                    data.total_blocked += stats.get("blocked", 0)
                    data.estimated_cost_drag_avoided += stats.get("cost_avoided", 0.0)

                    # Extract block reasons
                    reasons = stats.get("reasons", {})
                    data.blocked_by_interval += reasons.get("interval", 0)
                    data.blocked_by_daily_limit += reasons.get("daily_limit", 0)
                    data.blocked_by_ev_cost += reasons.get("ev_cost", 0)

                    # Estimate total decisions (blocked + allowed)
                    # We'll use shadow collection decisions if available
                    shadow = report.get("shadow_collection", {})
                    by_mode = shadow.get("decisions_by_mode", {})
                    paper_live = by_mode.get("PAPER_LIVE", 0)
                    # Rough estimate: distribute among tracked symbols
                    if len(per_symbol) > 0:
                        data.total_decisions += paper_live // max(1, len(per_symbol) - 1)

            except Exception as e:
                logger.warning(f"Could not load report {report_path}: {e}")

        # Also load current turnover state for additional context
        if self.turnover_state_path.exists():
            try:
                with open(self.turnover_state_path) as f:
                    state = json.load(f)

                for symbol_state in state.get("symbols", []):
                    symbol = symbol_state.get("symbol")
                    if symbol and symbol in symbol_data:
                        # Update with any additional data
                        pass

            except Exception as e:
                logger.warning(f"Could not load turnover state: {e}")

        # Update result summary
        result.symbols_analyzed = list(symbol_data.keys())
        result.total_decisions_week = sum(d.total_decisions for d in symbol_data.values())
        result.total_blocked_week = sum(d.total_blocked for d in symbol_data.values())
        result.total_cost_drag_avoided = sum(
            d.estimated_cost_drag_avoided for d in symbol_data.values()
        )

        logger.info(f"Loaded data for {len(symbol_data)} symbols")
        for symbol, data in symbol_data.items():
            logger.info(
                f"  {symbol}: decisions={data.total_decisions}, blocked={data.total_blocked}, "
                f"block_rate={data.block_rate:.1%}"
            )

        return symbol_data

    def _analyze_symbol(self, data: SymbolTuningData, result: TuningResult):
        """Analyze a symbol and generate recommendations."""
        symbol = data.symbol
        bounds = get_bounds_for_symbol(symbol)

        total_signals = data.total_decisions + data.total_blocked

        # Skip if not enough data
        if total_signals < self.MIN_DATA_POINTS:
            logger.info(f"  {symbol}: Skipping (only {total_signals} signals, need {self.MIN_DATA_POINTS})")
            return

        logger.info(f"  Analyzing {symbol}:")

        # Analyze interval blocking
        if data.interval_block_rate > self.HIGH_BLOCK_RATE:
            # Too many interval blocks - recommend loosening (reduce interval)
            new_interval = data.current_min_interval * 0.8  # 20% reduction
            new_interval = bounds.clamp_interval(new_interval)

            if new_interval < data.current_min_interval:
                result.recommendations.append(TuningRecommendation(
                    symbol=symbol,
                    parameter="min_interval",
                    current_value=data.current_min_interval,
                    recommended_value=new_interval,
                    reason=f"High interval block rate ({data.interval_block_rate:.1%}). Reducing interval to allow more trades.",
                    confidence=0.7,
                ))

        elif data.interval_block_rate < self.LOW_BLOCK_RATE and data.block_rate < self.LOW_BLOCK_RATE:
            # Very few blocks - recommend tightening (increase interval)
            new_interval = data.current_min_interval * 1.15  # 15% increase
            new_interval = bounds.clamp_interval(new_interval)

            if new_interval > data.current_min_interval:
                result.recommendations.append(TuningRecommendation(
                    symbol=symbol,
                    parameter="min_interval",
                    current_value=data.current_min_interval,
                    recommended_value=new_interval,
                    reason=f"Low block rate ({data.block_rate:.1%}). Increasing interval for better cost optimization.",
                    confidence=0.5,
                ))

        # Analyze daily limit blocking
        if data.daily_limit_block_rate > self.HIGH_BLOCK_RATE:
            # Too many daily limit blocks - recommend increasing limit
            new_max = data.current_max_decisions + 1
            new_max = bounds.clamp_max_decisions(new_max)

            if new_max > data.current_max_decisions:
                result.recommendations.append(TuningRecommendation(
                    symbol=symbol,
                    parameter="max_decisions",
                    current_value=float(data.current_max_decisions),
                    recommended_value=float(new_max),
                    reason=f"High daily limit block rate ({data.daily_limit_block_rate:.1%}). Increasing daily limit.",
                    confidence=0.6,
                ))

        elif data.daily_limit_block_rate == 0 and data.total_decisions < data.current_max_decisions * 0.5 * data.days_with_data:
            # Never hitting limit and under-utilizing - consider decreasing
            new_max = data.current_max_decisions - 1
            new_max = bounds.clamp_max_decisions(new_max)

            if new_max < data.current_max_decisions:
                result.recommendations.append(TuningRecommendation(
                    symbol=symbol,
                    parameter="max_decisions",
                    current_value=float(data.current_max_decisions),
                    recommended_value=float(new_max),
                    reason=f"Daily limit never reached and low utilization. Tightening limit.",
                    confidence=0.4,
                ))

        # Analyze EV/cost blocking with SAFETY RULES
        # CRITICAL: High EV/cost block rate should NOT automatically reduce EV multiple
        # This preserves trade quality. We only loosen OTHER parameters if needed.
        target_min = get_target_min_decisions_for_symbol(symbol)
        decisions_per_day = data.decisions_per_day

        if data.ev_cost_block_rate > self.HIGH_BLOCK_RATE:
            # High EV block rate - but do NOT reduce EV multiple by default
            # This is the SAFETY RULE to preserve trade quality

            rationale = TuningRationale(
                rule_applied="ev_cost_safety",
                decisions_per_day=decisions_per_day,
                target_min_decisions=target_min,
                ev_cost_block_rate=data.ev_cost_block_rate,
                action_taken="",  # Will be set below
                explanation="",
            )

            if decisions_per_day < target_min:
                # Decisions are BELOW target - consider loosening interval/max_decisions
                # but NOT the EV multiple
                rationale.action_taken = "loosen_interval_or_max_decisions"
                rationale.explanation = (
                    f"High EV/cost block rate ({data.ev_cost_block_rate:.1%}) with decisions/day "
                    f"({decisions_per_day:.1f}) below target ({target_min}). "
                    "May loosen interval or max_decisions, but NOT EV multiple to preserve quality."
                )

                # Check if we can loosen interval (if not already at minimum)
                if data.current_min_interval > bounds.min_interval_min:
                    new_interval = data.current_min_interval * 0.85  # 15% reduction
                    new_interval = bounds.clamp_interval(new_interval)

                    if new_interval < data.current_min_interval:
                        result.recommendations.append(TuningRecommendation(
                            symbol=symbol,
                            parameter="min_interval",
                            current_value=data.current_min_interval,
                            recommended_value=new_interval,
                            reason=(
                                f"EV/cost block rate high ({data.ev_cost_block_rate:.1%}) "
                                f"and decisions/day ({decisions_per_day:.1f}) < target ({target_min}). "
                                "Loosening interval to allow more decisions while preserving EV quality."
                            ),
                            confidence=0.6,
                            tuning_rationale=rationale,
                        ))

                # Check if we can increase max_decisions (if not already at maximum)
                elif data.current_max_decisions < bounds.max_decisions_max:
                    new_max = data.current_max_decisions + 1
                    new_max = bounds.clamp_max_decisions(new_max)

                    if new_max > data.current_max_decisions:
                        result.recommendations.append(TuningRecommendation(
                            symbol=symbol,
                            parameter="max_decisions",
                            current_value=float(data.current_max_decisions),
                            recommended_value=float(new_max),
                            reason=(
                                f"EV/cost block rate high ({data.ev_cost_block_rate:.1%}) "
                                f"and decisions/day ({decisions_per_day:.1f}) < target ({target_min}). "
                                "Increasing max decisions while preserving EV quality."
                            ),
                            confidence=0.5,
                            tuning_rationale=rationale,
                        ))
            else:
                # Decisions are AT or ABOVE target - do NOT change anything
                # High EV block rate is ACCEPTABLE because we're meeting decision targets
                rationale.action_taken = "no_change"
                rationale.explanation = (
                    f"High EV/cost block rate ({data.ev_cost_block_rate:.1%}) but decisions/day "
                    f"({decisions_per_day:.1f}) >= target ({target_min}). "
                    "EV multiple PRESERVED to maintain trade quality. No changes needed."
                )
                logger.info(
                    f"    EV/cost safety: block_rate={data.ev_cost_block_rate:.1%}, "
                    f"decisions/day={decisions_per_day:.1f}, target={target_min} -> NO CHANGE"
                )

        elif data.ev_cost_block_rate < self.LOW_BLOCK_RATE * 0.5:
            # Very few EV blocks - recommend TIGHTENING (increase EV multiple)
            # This is SAFE because it improves trade quality
            new_ev = data.current_ev_multiple * 1.1  # 10% increase
            new_ev = bounds.clamp_ev_multiple(new_ev)

            if new_ev > data.current_ev_multiple:
                rationale = TuningRationale(
                    rule_applied="ev_cost_tightening",
                    decisions_per_day=decisions_per_day,
                    target_min_decisions=target_min,
                    ev_cost_block_rate=data.ev_cost_block_rate,
                    action_taken="increase_ev_multiple",
                    explanation=(
                        f"Low EV/cost block rate ({data.ev_cost_block_rate:.1%}) indicates "
                        "room to increase quality bar. Tightening EV requirement."
                    ),
                )
                result.recommendations.append(TuningRecommendation(
                    symbol=symbol,
                    parameter="ev_multiple",
                    current_value=data.current_ev_multiple,
                    recommended_value=new_ev,
                    reason=f"Low EV block rate ({data.ev_cost_block_rate:.1%}). Increasing EV requirement for better signal quality.",
                    confidence=0.5,
                    tuning_rationale=rationale,
                ))

        logger.info(f"    Current: interval={data.current_min_interval}, max_daily={data.current_max_decisions}, ev={data.current_ev_multiple}")
        logger.info(f"    Block rates: interval={data.interval_block_rate:.1%}, daily={data.daily_limit_block_rate:.1%}, ev={data.ev_cost_block_rate:.1%}")

    def _apply_changes(self, result: TuningResult):
        """Apply recommended changes with rollback support."""
        if not result.recommendations:
            return

        try:
            from bot.turnover_governor import (
                TurnoverGovernor,
                TurnoverGovernorConfig,
                SymbolOverrideConfig,
                get_turnover_governor,
                reset_turnover_governor,
            )

            # Get current governor
            governor = get_turnover_governor()

            # Save current config as rollback
            rollback_data = {
                "timestamp": datetime.now().isoformat(),
                "reason": "Pre-tuning backup",
                "config": {
                    "min_decision_interval_minutes": governor.config.min_decision_interval_minutes,
                    "max_decisions_per_day": governor.config.max_decisions_per_day,
                    "min_expected_value_multiple": governor.config.min_expected_value_multiple,
                    "symbol_overrides": {
                        pattern: override.to_dict()
                        for pattern, override in governor.config.symbol_overrides.items()
                    },
                },
            }

            result.previous_config = rollback_data["config"]

            # Save rollback file
            rollback_path = self.output_dir / f"turnover_config_rollback_{datetime.now().strftime('%Y-%m-%d')}.json"
            with open(rollback_path, "w") as f:
                json.dump(rollback_data, f, indent=2)
            result.rollback_file = str(rollback_path)

            logger.info(f"Saved rollback config to {rollback_path}")

            # Apply changes
            for rec in result.recommendations:
                symbol = rec.symbol

                # Get or create override for this symbol
                if symbol not in governor.config.symbol_overrides:
                    governor.config.symbol_overrides[symbol] = SymbolOverrideConfig()

                override = governor.config.symbol_overrides[symbol]

                if rec.parameter == "min_interval":
                    override.min_interval_minutes = rec.recommended_value
                    logger.info(f"Applied {symbol} min_interval: {rec.current_value} -> {rec.recommended_value}")
                elif rec.parameter == "max_decisions":
                    override.max_decisions_per_day = int(rec.recommended_value)
                    logger.info(f"Applied {symbol} max_decisions: {int(rec.current_value)} -> {int(rec.recommended_value)}")
                elif rec.parameter == "ev_multiple":
                    override.min_ev_cost_multiple = rec.recommended_value
                    logger.info(f"Applied {symbol} ev_multiple: {rec.current_value} -> {rec.recommended_value}")

            # Save updated state
            governor._save_state()
            result.recommendations_applied = True

            logger.info("All recommendations applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply changes: {e}")
            result.recommendations_applied = False

    def _save_result(self, result: TuningResult):
        """Save tuning result as audit artifact."""
        output_path = self.output_dir / f"turnover_tuning_{datetime.now().strftime('%Y-%m-%d')}.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved tuning result to {output_path}")

    @staticmethod
    def rollback(rollback_file: str) -> bool:
        """
        Rollback to a previous configuration.

        Args:
            rollback_file: Path to rollback JSON file

        Returns:
            True if rollback succeeded
        """
        try:
            from bot.turnover_governor import (
                TurnoverGovernorConfig,
                SymbolOverrideConfig,
                get_turnover_governor,
                reset_turnover_governor,
            )

            with open(rollback_file) as f:
                rollback_data = json.load(f)

            config = rollback_data.get("config", {})

            # Reset singleton to apply new config
            reset_turnover_governor()

            # Create new config with rollback values
            symbol_overrides = {}
            for pattern, override_data in config.get("symbol_overrides", {}).items():
                symbol_overrides[pattern] = SymbolOverrideConfig.from_dict(override_data)

            new_config = TurnoverGovernorConfig(
                min_decision_interval_minutes=config.get("min_decision_interval_minutes", 15.0),
                max_decisions_per_day=config.get("max_decisions_per_day", 10),
                min_expected_value_multiple=config.get("min_expected_value_multiple", 2.0),
                symbol_overrides=symbol_overrides,
            )

            # Get governor with new config
            governor = get_turnover_governor(new_config)

            logger.info(f"Successfully rolled back to config from {rollback_data.get('timestamp', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run weekly turnover tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="Weekly Turnover Governor Tuner")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply recommended changes (default: dry-run only)",
    )
    parser.add_argument(
        "--rollback",
        type=str,
        help="Rollback to a previous config file",
    )
    args = parser.parse_args()

    if args.rollback:
        logger.info(f"Rolling back to: {args.rollback}")
        success = WeeklyTurnoverTuner.rollback(args.rollback)
        sys.exit(0 if success else 1)

    tuner = WeeklyTurnoverTuner()
    result = tuner.run_tuning(apply_changes=args.apply)

    # Print summary
    print("\n" + "=" * 60)
    print("WEEKLY TURNOVER TUNING RESULT")
    print("=" * 60)
    print(f"\nPeriod: {result.week_start} to {result.week_end}")
    print(f"Safety Gate: {'PASSED' if result.safety_gate_passed else 'FAILED'}")
    print(f"  Reason: {result.safety_gate_reason}")
    print(f"  PAPER_LIVE weeks: {result.paper_live_weeks_counted}")

    if result.critical_alerts_found:
        print(f"\nCritical Alerts Found:")
        for alert in result.critical_alerts_found:
            print(f"  - {alert}")

    print(f"\nData Summary:")
    print(f"  Symbols analyzed: {', '.join(result.symbols_analyzed) or 'None'}")
    print(f"  Total decisions: {result.total_decisions_week}")
    print(f"  Total blocked: {result.total_blocked_week}")
    print(f"  Cost drag avoided: ${result.total_cost_drag_avoided:.2f}")

    if result.recommendations:
        print(f"\nRecommendations ({len(result.recommendations)}):")
        for rec in result.recommendations:
            change_dir = "↑" if rec.recommended_value > rec.current_value else "↓"
            print(f"  {rec.symbol} {rec.parameter}: {rec.current_value} -> {rec.recommended_value} {change_dir}")
            print(f"    Reason: {rec.reason}")
            print(f"    Confidence: {rec.confidence:.0%}")
    else:
        print("\nNo recommendations generated.")

    print(f"\nChanges applied: {result.recommendations_applied}")
    if result.rollback_file:
        print(f"Rollback file: {result.rollback_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()
