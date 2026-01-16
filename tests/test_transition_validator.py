"""
Tests for transition validator module.
"""

import pytest
from datetime import datetime, timedelta

from bot.trading_mode import TradingMode, ModeState, TradingStatus
from bot.transition_validator import (
    TransitionProgress,
    TransitionResult,
    TransitionValidator,
    create_transition_validator,
)


class TestTransitionProgress:
    """Test TransitionProgress dataclass."""

    def test_progress_creation(self):
        """Test creating progress instance."""
        progress = TransitionProgress(
            requirement="Days in current mode",
            current=10,
            required=14,
            passed=False,
            progress_pct=71.4,
        )
        assert progress.requirement == "Days in current mode"
        assert progress.current == 10
        assert progress.required == 14
        assert progress.passed is False

    def test_passed_progress(self):
        """Test progress that has passed."""
        progress = TransitionProgress(
            requirement="Total trades",
            current=150,
            required=100,
            passed=True,
            progress_pct=100.0,
        )
        assert progress.passed is True
        assert progress.progress_pct == 100.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        progress = TransitionProgress(
            requirement="Win rate",
            current=50.0,
            required=45.0,
            passed=True,
            progress_pct=100.0,
        )
        d = progress.to_dict()

        assert d["requirement"] == "Win rate"
        assert d["current"] == 50.0
        assert d["required"] == 45.0
        assert d["passed"] is True
        assert d["progress_pct"] == 100.0


class TestTransitionResult:
    """Test TransitionResult dataclass."""

    def test_result_creation(self):
        """Test creating result instance."""
        result = TransitionResult(
            allowed=True,
            from_mode=TradingMode.PAPER_LIVE_DATA,
            to_mode=TradingMode.TESTNET,
            progress=[],
            blocking_reasons=[],
            requires_approval=True,
        )
        assert result.allowed is True
        assert result.from_mode == TradingMode.PAPER_LIVE_DATA
        assert result.to_mode == TradingMode.TESTNET

    def test_blocked_result(self):
        """Test blocked result with reasons."""
        progress = TransitionProgress(
            requirement="Days in current mode",
            current=5,
            required=14,
            passed=False,
            progress_pct=35.7,
        )
        result = TransitionResult(
            allowed=False,
            from_mode=TradingMode.PAPER_LIVE_DATA,
            to_mode=TradingMode.TESTNET,
            progress=[progress],
            blocking_reasons=["Need 14 days in mode, have 5"],
            requires_approval=True,
        )
        assert result.allowed is False
        assert len(result.blocking_reasons) == 1
        assert len(result.progress) == 1

    def test_approved_result(self):
        """Test approved result."""
        result = TransitionResult(
            allowed=True,
            from_mode=TradingMode.TESTNET,
            to_mode=TradingMode.LIVE_LIMITED,
            progress=[],
            blocking_reasons=[],
            requires_approval=True,
            approved=True,
            approved_by="admin",
            approved_at="2024-01-15T10:30:00",
        )
        assert result.approved is True
        assert result.approved_by == "admin"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TransitionResult(
            allowed=True,
            from_mode=TradingMode.PAPER_LIVE_DATA,
            to_mode=TradingMode.TESTNET,
            progress=[],
            blocking_reasons=[],
            requires_approval=False,
        )
        d = result.to_dict()

        assert d["allowed"] is True
        assert d["from_mode"] == "paper_live_data"
        assert d["to_mode"] == "testnet"
        assert d["requires_approval"] is False


class TestTransitionValidator:
    """Test TransitionValidator class."""

    @pytest.fixture
    def validator(self):
        """Create transition validator."""
        return TransitionValidator()

    @pytest.fixture
    def good_state(self):
        """Create state that meets all requirements."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now() - timedelta(days=20)
        state.total_trades = 150
        state.winning_trades = 80
        state.losing_trades = 70
        state.total_pnl = 500.0
        state.max_drawdown_pct = 0.08
        return state

    @pytest.fixture
    def poor_state(self):
        """Create state that doesn't meet requirements."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now() - timedelta(days=3)
        state.total_trades = 20
        state.winning_trades = 5
        state.losing_trades = 15
        state.total_pnl = -100.0
        state.max_drawdown_pct = 0.20
        return state

    def test_validator_creation(self, validator):
        """Test validator is created."""
        assert validator is not None
        assert validator._pending_approvals == {}

    def test_can_transition_good_state(self, validator, good_state):
        """Test transition with good state."""
        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )
        # Should be allowed (meets all requirements)
        assert result.allowed is True
        assert len(result.blocking_reasons) == 0

    def test_can_transition_poor_state(self, validator, poor_state):
        """Test transition with poor state."""
        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            poor_state,
        )
        # Should be blocked
        assert result.allowed is False
        assert len(result.blocking_reasons) > 0

    def test_cannot_skip_modes(self, validator, good_state):
        """Test cannot skip modes in progression."""
        result = validator.can_transition(
            TradingMode.PAPER_SYNTHETIC,
            TradingMode.TESTNET,
            good_state,
        )
        # Cannot skip PAPER_LIVE_DATA
        assert result.allowed is False
        assert any("Cannot transition directly" in r for r in result.blocking_reasons)

    def test_backward_transition_allowed(self, validator, poor_state):
        """Test backward transition is always allowed."""
        poor_state.mode = TradingMode.TESTNET
        result = validator.can_transition(
            TradingMode.TESTNET,
            TradingMode.PAPER_LIVE_DATA,
            poor_state,
        )
        # Backward is always allowed
        assert result.allowed is True
        assert len(result.blocking_reasons) == 0

    def test_days_requirement_check(self, validator):
        """Test days requirement check."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now() - timedelta(days=5)
        state.total_trades = 200
        state.winning_trades = 100
        state.total_pnl = 1000.0

        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            state,
        )
        # Should fail days requirement
        days_progress = next(p for p in result.progress if "Days" in p.requirement)
        assert days_progress.passed is False

    def test_trades_requirement_check(self, validator):
        """Test trades requirement check."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now() - timedelta(days=20)
        state.total_trades = 30  # Below 100
        state.winning_trades = 20
        state.total_pnl = 500.0

        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            state,
        )
        # Should fail trades requirement
        trades_progress = next(p for p in result.progress if "trades" in p.requirement.lower())
        assert trades_progress.passed is False

    def test_win_rate_requirement_check(self, validator):
        """Test win rate requirement check."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now() - timedelta(days=20)
        state.total_trades = 150
        state.winning_trades = 50  # 33% win rate < 45%
        state.losing_trades = 100
        state.total_pnl = 100.0

        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            state,
        )
        # Should fail win rate
        assert result.allowed is False
        assert any("win rate" in r.lower() for r in result.blocking_reasons)

    def test_drawdown_requirement_check(self, validator):
        """Test drawdown requirement check."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now() - timedelta(days=20)
        state.total_trades = 150
        state.winning_trades = 80
        state.total_pnl = 500.0
        state.max_drawdown_pct = 0.20  # > 12%

        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            state,
        )
        # Should fail drawdown
        assert any("drawdown" in r.lower() for r in result.blocking_reasons)

    def test_get_transition_progress(self, validator, good_state):
        """Test getting transition progress."""
        progress = validator.get_transition_progress(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )

        assert "overall_progress" in progress
        assert "allowed" in progress
        assert "progress_details" in progress
        assert "blocking_reasons" in progress
        assert "estimated_days_remaining" in progress

    def test_progress_percentage_calculation(self, validator, poor_state):
        """Test overall progress calculation."""
        progress = validator.get_transition_progress(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            poor_state,
        )

        # Should have progress less than 100%
        assert progress["overall_progress"] < 100
        assert len(progress["progress_details"]) > 0

    def test_estimate_days_remaining(self, validator, poor_state):
        """Test days remaining estimation."""
        progress = validator.get_transition_progress(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            poor_state,
        )

        # Should estimate days remaining
        assert progress["estimated_days_remaining"] is not None or progress["allowed"]

    def test_request_approval(self, validator, good_state):
        """Test requesting approval."""
        approval_id, result = validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )

        # Should require approval for this transition
        assert result.requires_approval is True
        assert approval_id != ""
        assert approval_id in validator._pending_approvals

    def test_request_approval_blocked(self, validator, poor_state):
        """Test requesting approval when blocked."""
        approval_id, result = validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            poor_state,
        )

        # Should not create approval when blocked
        assert result.allowed is False
        assert approval_id == ""

    def test_approve_transition(self, validator, good_state):
        """Test approving a transition."""
        approval_id, _ = validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )

        success, message = validator.approve_transition(approval_id, "admin")

        assert success is True
        assert "admin" in message
        assert validator._pending_approvals[approval_id].approved is True

    def test_approve_nonexistent(self, validator):
        """Test approving nonexistent approval."""
        success, message = validator.approve_transition("fake_id", "admin")

        assert success is False
        assert "not found" in message

    def test_reject_transition(self, validator, good_state):
        """Test rejecting a transition."""
        approval_id, _ = validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )

        success, message = validator.reject_transition(approval_id, "Too risky")

        assert success is True
        assert "rejected" in message.lower()
        assert approval_id not in validator._pending_approvals

    def test_reject_nonexistent(self, validator):
        """Test rejecting nonexistent approval."""
        success, message = validator.reject_transition("fake_id", "reason")

        assert success is False
        assert "not found" in message

    def test_get_pending_approvals(self, validator, good_state):
        """Test getting pending approvals."""
        validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )

        pending = validator.get_pending_approvals()

        assert len(pending) == 1
        assert "id" in pending[0]
        assert "from_mode" in pending[0]
        assert "to_mode" in pending[0]

    def test_is_approved_pending(self, validator, good_state):
        """Test checking if approval is pending."""
        approval_id, _ = validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )

        approved, result = validator.is_approved(approval_id)

        assert approved is False
        assert result is not None

    def test_is_approved_approved(self, validator, good_state):
        """Test checking if approval is approved."""
        approval_id, _ = validator.request_approval(
            TradingMode.PAPER_LIVE_DATA,
            TradingMode.TESTNET,
            good_state,
        )
        validator.approve_transition(approval_id, "admin")

        approved, result = validator.is_approved(approval_id)

        assert approved is True
        assert result.approved is True
        # Should be removed from pending after checking
        assert approval_id not in validator._pending_approvals

    def test_is_approved_nonexistent(self, validator):
        """Test checking nonexistent approval."""
        approved, result = validator.is_approved("fake_id")

        assert approved is False
        assert result is None

    def test_get_recommended_next_mode(self, validator, good_state):
        """Test getting recommended next mode."""
        recommended = validator.get_recommended_next_mode(
            TradingMode.PAPER_LIVE_DATA,
            good_state,
        )

        assert recommended == TradingMode.TESTNET

    def test_get_recommended_next_mode_not_ready(self, validator, poor_state):
        """Test no recommendation when not ready."""
        recommended = validator.get_recommended_next_mode(
            TradingMode.PAPER_LIVE_DATA,
            poor_state,
        )

        assert recommended is None

    def test_get_recommended_at_highest_mode(self, validator, good_state):
        """Test no recommendation at highest mode."""
        good_state.mode = TradingMode.LIVE_FULL
        recommended = validator.get_recommended_next_mode(
            TradingMode.LIVE_FULL,
            good_state,
        )

        assert recommended is None

    def test_should_downgrade_severe_drawdown(self, validator):
        """Test downgrade recommendation on severe drawdown."""
        state = ModeState(mode=TradingMode.LIVE_LIMITED)
        state.max_drawdown_pct = 0.20  # 20% > 15%

        should, mode, reason = validator.should_downgrade(
            TradingMode.LIVE_LIMITED,
            state,
        )

        assert should is True
        assert mode == TradingMode.TESTNET
        assert "drawdown" in reason.lower()

    def test_should_downgrade_consecutive_losses(self, validator):
        """Test downgrade recommendation on consecutive losses."""
        state = ModeState(mode=TradingMode.LIVE_LIMITED)
        state.consecutive_losses = 12

        should, mode, reason = validator.should_downgrade(
            TradingMode.LIVE_LIMITED,
            state,
        )

        assert should is True
        assert mode == TradingMode.PAPER_LIVE_DATA
        assert "consecutive" in reason.lower()

    def test_should_downgrade_poor_win_rate(self, validator):
        """Test downgrade recommendation on poor win rate."""
        state = ModeState(mode=TradingMode.LIVE_LIMITED)
        state.total_trades = 100
        state.winning_trades = 20  # 20% < 30%
        state.losing_trades = 80

        should, mode, reason = validator.should_downgrade(
            TradingMode.LIVE_LIMITED,
            state,
        )

        assert should is True
        assert mode == TradingMode.PAPER_LIVE_DATA
        assert "win rate" in reason.lower()

    def test_should_not_downgrade_paper_mode(self, validator):
        """Test no downgrade for paper mode."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.max_drawdown_pct = 0.30

        should, mode, reason = validator.should_downgrade(
            TradingMode.PAPER_LIVE_DATA,
            state,
        )

        assert should is False
        assert mode is None

    def test_should_not_downgrade_good_performance(self, validator, good_state):
        """Test no downgrade for good performance."""
        good_state.mode = TradingMode.LIVE_LIMITED

        should, mode, reason = validator.should_downgrade(
            TradingMode.LIVE_LIMITED,
            good_state,
        )

        assert should is False
        assert mode is None
        assert reason == ""


class TestCreateTransitionValidator:
    """Test factory function."""

    def test_create_validator(self):
        """Test creating validator via factory."""
        validator = create_transition_validator()

        assert validator is not None
        assert isinstance(validator, TransitionValidator)
