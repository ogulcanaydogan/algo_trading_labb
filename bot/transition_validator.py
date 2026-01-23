"""
Transition Validator Module.

Validates readiness to transition between trading modes.
Implements strict requirements to prevent premature live trading.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bot.trading_mode import (
    ModeState,
    TradingMode,
    TransitionRequirements,
    get_transition_requirements,
)

logger = logging.getLogger(__name__)


@dataclass
class TransitionProgress:
    """Progress towards transition requirements."""

    requirement: str
    current: float
    required: float
    passed: bool
    progress_pct: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "requirement": self.requirement,
            "current": self.current,
            "required": self.required,
            "passed": self.passed,
            "progress_pct": self.progress_pct,
        }


@dataclass
class TransitionResult:
    """Result of a transition validation."""

    allowed: bool
    from_mode: TradingMode
    to_mode: TradingMode
    progress: List[TransitionProgress]
    blocking_reasons: List[str]
    requires_approval: bool
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "progress": [p.to_dict() for p in self.progress],
            "blocking_reasons": self.blocking_reasons,
            "requires_approval": self.requires_approval,
            "approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }


class TransitionValidator:
    """
    Validates transitions between trading modes.

    Implements strict requirements (user selected):
    - 14+ days paper trading before testnet/live
    - 100+ trades minimum
    - 45%+ win rate
    - Profit factor > 1.0
    - Max 10-12% drawdown
    """

    def __init__(self):
        self._pending_approvals: Dict[str, TransitionResult] = {}
        self._user_preferences: Dict[str, Any] = {}

    def set_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Set user preferences for transition behavior."""
        self._user_preferences = preferences

    def get_user_preferences(self) -> Dict[str, Any]:
        """Get current user preferences."""
        return self._user_preferences

    def can_transition(
        self, from_mode: TradingMode, to_mode: TradingMode, state: ModeState
    ) -> TransitionResult:
        """
        Check if transition is allowed.

        Returns detailed progress and blocking reasons.
        """
        progress: List[TransitionProgress] = []
        blocking_reasons: List[str] = []

        # Check if direct transition is allowed
        if not from_mode.can_transition_to(to_mode):
            blocking_reasons.append(
                f"Cannot transition directly from {from_mode.value} to {to_mode.value}"
            )
            return TransitionResult(
                allowed=False,
                from_mode=from_mode,
                to_mode=to_mode,
                progress=progress,
                blocking_reasons=blocking_reasons,
                requires_approval=True,
            )

        # Get requirements
        requirements = get_transition_requirements(from_mode, to_mode)

        if requirements is None:
            # No specific requirements (e.g., going backward)
            progression = TradingMode.get_progression()
            from_idx = progression.index(from_mode)
            to_idx = progression.index(to_mode)
            if to_idx < from_idx:
                # Going backward is always allowed
                return TransitionResult(
                    allowed=True,
                    from_mode=from_mode,
                    to_mode=to_mode,
                    progress=[],
                    blocking_reasons=[],
                    requires_approval=False,
                )
            blocking_reasons.append("No transition path defined")
            return TransitionResult(
                allowed=False,
                from_mode=from_mode,
                to_mode=to_mode,
                progress=progress,
                blocking_reasons=blocking_reasons,
                requires_approval=True,
            )

        # Check days in mode
        days = state.days_in_mode
        days_required = requirements.min_days_in_current_mode
        days_passed = days >= days_required
        progress.append(
            TransitionProgress(
                requirement="Days in current mode",
                current=days,
                required=days_required,
                passed=days_passed,
                progress_pct=min(100, days / days_required * 100) if days_required > 0 else 100,
            )
        )
        if not days_passed:
            blocking_reasons.append(f"Need {days_required} days in mode, have {days}")

        # Check total trades
        trades = state.total_trades
        trades_required = requirements.min_trades
        trades_passed = trades >= trades_required
        progress.append(
            TransitionProgress(
                requirement="Total trades",
                current=trades,
                required=trades_required,
                passed=trades_passed,
                progress_pct=min(100, trades / trades_required * 100)
                if trades_required > 0
                else 100,
            )
        )
        if not trades_passed:
            blocking_reasons.append(f"Need {trades_required} trades, have {trades}")

        # Check win rate
        win_rate = state.win_rate
        win_rate_required = requirements.min_win_rate
        win_rate_passed = win_rate >= win_rate_required
        progress.append(
            TransitionProgress(
                requirement="Win rate",
                current=win_rate * 100,
                required=win_rate_required * 100,
                passed=win_rate_passed,
                progress_pct=min(100, win_rate / win_rate_required * 100)
                if win_rate_required > 0
                else 100,
            )
        )
        if not win_rate_passed:
            blocking_reasons.append(f"Need {win_rate_required:.0%} win rate, have {win_rate:.0%}")

        # Check max drawdown
        drawdown = state.max_drawdown_pct
        max_drawdown = requirements.max_drawdown_pct
        drawdown_passed = drawdown <= max_drawdown
        progress.append(
            TransitionProgress(
                requirement="Max drawdown (lower is better)",
                current=drawdown * 100,
                required=max_drawdown * 100,
                passed=drawdown_passed,
                progress_pct=100
                if drawdown_passed
                else (max_drawdown / drawdown * 100)
                if drawdown > 0
                else 100,
            )
        )
        if not drawdown_passed:
            blocking_reasons.append(f"Drawdown {drawdown:.1%} exceeds max {max_drawdown:.1%}")

        # Check profit factor
        profit_factor = state.profit_factor
        pf_required = requirements.min_profit_factor
        pf_passed = profit_factor >= pf_required
        progress.append(
            TransitionProgress(
                requirement="Profit factor",
                current=profit_factor,
                required=pf_required,
                passed=pf_passed,
                progress_pct=min(100, profit_factor / pf_required * 100)
                if pf_required > 0
                else 100,
            )
        )
        if not pf_passed:
            blocking_reasons.append(f"Need profit factor {pf_required}, have {profit_factor:.2f}")

        # Apply user preferences for flexible transitions
        if self._user_preferences:
            # If user has set a custom tolerance, adjust requirements
            tolerance = self._user_preferences.get("transition_tolerance", 0.0)
            if tolerance > 0:
                # Allow transition if user has specified tolerance for relaxed requirements
                # This is a simple implementation - in practice, you might want more nuanced logic
                pass

        # Determine if allowed
        all_passed = len(blocking_reasons) == 0

        return TransitionResult(
            allowed=all_passed,
            from_mode=from_mode,
            to_mode=to_mode,
            progress=progress,
            blocking_reasons=blocking_reasons,
            requires_approval=requirements.require_manual_approval,
        )

    def get_transition_progress(
        self, from_mode: TradingMode, to_mode: TradingMode, state: ModeState
    ) -> Dict[str, Any]:
        """
        Get detailed progress towards a transition.

        Returns a summary suitable for dashboard display.
        """
        result = self.can_transition(from_mode, to_mode, state)

        # Calculate overall progress
        if result.progress:
            overall_progress = sum(p.progress_pct for p in result.progress) / len(result.progress)
        else:
            overall_progress = 100 if result.allowed else 0

        return {
            "from_mode": from_mode.value,
            "to_mode": to_mode.value,
            "overall_progress": overall_progress,
            "allowed": result.allowed,
            "requires_approval": result.requires_approval,
            "progress_details": [p.to_dict() for p in result.progress],
            "blocking_reasons": result.blocking_reasons,
            "estimated_days_remaining": self._estimate_days_remaining(result, state),
        }

    def _estimate_days_remaining(self, result: TransitionResult, state: ModeState) -> Optional[int]:
        """Estimate days until transition is possible."""
        if result.allowed:
            return 0

        # Find the blocking requirement that will take longest
        max_days = 0

        for progress in result.progress:
            if not progress.passed:
                if progress.requirement == "Days in current mode":
                    days_needed = int(progress.required - progress.current)
                    max_days = max(max_days, days_needed)
                elif progress.requirement == "Total trades":
                    # Estimate based on current trading rate
                    if state.days_in_mode > 0:
                        trades_per_day = state.total_trades / state.days_in_mode
                        if trades_per_day > 0:
                            trades_needed = progress.required - progress.current
                            days_needed = int(trades_needed / trades_per_day)
                            max_days = max(max_days, days_needed)

        return max_days if max_days > 0 else None

    def request_approval(
        self, from_mode: TradingMode, to_mode: TradingMode, state: ModeState
    ) -> Tuple[str, TransitionResult]:
        """
        Request approval for a transition.

        Returns (approval_id, result).
        """
        result = self.can_transition(from_mode, to_mode, state)

        if not result.allowed:
            return "", result

        if not result.requires_approval:
            result.approved = True
            return "", result

        # Generate approval ID
        approval_id = (
            f"approval_{from_mode.value}_{to_mode.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self._pending_approvals[approval_id] = result

        logger.info(
            f"Transition approval requested: {from_mode.value} -> {to_mode.value} "
            f"(ID: {approval_id})"
        )

        return approval_id, result

    def approve_transition(self, approval_id: str, approver: str) -> Tuple[bool, str]:
        """
        Approve a pending transition.

        Returns (success, message).
        """
        if approval_id not in self._pending_approvals:
            return False, f"Approval ID {approval_id} not found"

        result = self._pending_approvals[approval_id]
        result.approved = True
        result.approved_by = approver
        result.approved_at = datetime.now().isoformat()

        logger.warning(
            f"Transition APPROVED: {result.from_mode.value} -> {result.to_mode.value} by {approver}"
        )

        return True, f"Transition approved by {approver}"

    def reject_transition(self, approval_id: str, reason: str) -> Tuple[bool, str]:
        """
        Reject a pending transition.

        Returns (success, message).
        """
        if approval_id not in self._pending_approvals:
            return False, f"Approval ID {approval_id} not found"

        result = self._pending_approvals.pop(approval_id)

        logger.info(
            f"Transition REJECTED: {result.from_mode.value} -> {result.to_mode.value} "
            f"reason: {reason}"
        )

        return True, f"Transition rejected: {reason}"

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get list of pending approvals."""
        return [
            {
                "id": k,
                "from_mode": v.from_mode.value,
                "to_mode": v.to_mode.value,
                "progress": [p.to_dict() for p in v.progress],
            }
            for k, v in self._pending_approvals.items()
        ]

    def is_approved(self, approval_id: str) -> Tuple[bool, Optional[TransitionResult]]:
        """Check if a transition has been approved."""
        if approval_id not in self._pending_approvals:
            return False, None

        result = self._pending_approvals.get(approval_id)
        if result and result.approved:
            # Remove from pending after use
            self._pending_approvals.pop(approval_id, None)
            return True, result

        return False, result

    def get_recommended_next_mode(
        self, current_mode: TradingMode, state: ModeState
    ) -> Optional[TradingMode]:
        """
        Get the recommended next mode based on current performance.

        Returns None if no transition is recommended.
        """
        progression = TradingMode.get_progression()
        current_idx = progression.index(current_mode)

        if current_idx >= len(progression) - 1:
            return None  # Already at highest mode

        next_mode = progression[current_idx + 1]
        result = self.can_transition(current_mode, next_mode, state)

        if result.allowed:
            return next_mode

        # If not allowed by strict requirements, check if user preferences allow it
        if self._user_preferences and self._user_preferences.get(
            "allow_riskier_transitions", False
        ):
            # Allow transition with warning if user has explicitly allowed riskier transitions
            return next_mode

        return None

    def should_downgrade(
        self, current_mode: TradingMode, state: ModeState
    ) -> Tuple[bool, Optional[TradingMode], str]:
        """
        Check if trading should be downgraded to a safer mode.

        Returns (should_downgrade, recommended_mode, reason).
        """
        # Check for severe drawdown
        if state.max_drawdown_pct > 0.15:  # 15% drawdown
            if current_mode.is_live:
                return (
                    True,
                    TradingMode.TESTNET,
                    f"Severe drawdown: {state.max_drawdown_pct:.1%}",
                )

        # Check for too many consecutive losses
        if state.consecutive_losses >= 10:
            if current_mode.is_live:
                return (
                    True,
                    TradingMode.PAPER_LIVE_DATA,
                    f"Too many consecutive losses: {state.consecutive_losses}",
                )

        # Check for poor win rate over significant trades
        if state.total_trades >= 50 and state.win_rate < 0.30:
            if current_mode.is_live:
                return (
                    True,
                    TradingMode.PAPER_LIVE_DATA,
                    f"Poor win rate: {state.win_rate:.1%}",
                )

        # Apply user preferences for downgrade behavior
        if self._user_preferences:
            # If user has set a custom downgrade threshold, use that
            custom_downgrade_threshold = self._user_preferences.get("downgrade_threshold")
            if custom_downgrade_threshold is not None:
                if state.max_drawdown_pct > custom_downgrade_threshold:
                    return (
                        True,
                        TradingMode.PAPER_LIVE_DATA,
                        f"Custom downgrade threshold triggered: {state.max_drawdown_pct:.1%}",
                    )

        return False, None, ""


def create_transition_validator() -> TransitionValidator:
    """Create a transition validator instance."""
    validator = TransitionValidator()
    # Set default preferences
    validator.set_user_preferences(
        {
            "allow_riskier_transitions": False,
            "transition_tolerance": 0.0,
            "downgrade_threshold": 0.10,  # 10% drawdown threshold for automatic downgrade
        }
    )
    return validator
