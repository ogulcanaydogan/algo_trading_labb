"""Tests for bot.control module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot.control import (
    BotControlState,
    load_bot_control,
    save_bot_control,
    update_bot_control,
    _control_path,
    _utcnow,
)


class TestBotControlState:
    """Tests for BotControlState dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        state = BotControlState()

        assert state.paused is False
        assert state.reason is None
        assert state.updated_at is None

    def test_custom_values(self) -> None:
        """Test creating with custom values."""
        now = _utcnow()
        state = BotControlState(
            paused=True,
            reason="Manual stop",
            updated_at=now,
        )

        assert state.paused is True
        assert state.reason == "Manual stop"
        assert state.updated_at == now

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal data."""
        state = BotControlState(paused=True, reason=None)

        result = state.to_dict()

        assert result["paused"] is True
        assert result["reason"] is None
        assert "updated_at" not in result

    def test_to_dict_full(self) -> None:
        """Test to_dict with all data."""
        now = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        state = BotControlState(
            paused=True,
            reason="Maintenance",
            updated_at=now,
        )

        result = state.to_dict()

        assert result["paused"] is True
        assert result["reason"] == "Maintenance"
        assert result["updated_at"] == "2024-01-15T10:30:00+00:00"

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal data."""
        payload = {"paused": True}

        state = BotControlState.from_dict(payload)

        assert state.paused is True
        assert state.reason is None
        assert state.updated_at is None

    def test_from_dict_full(self) -> None:
        """Test from_dict with all data."""
        payload = {
            "paused": True,
            "reason": "Test",
            "updated_at": "2024-01-15T10:30:00+00:00",
        }

        state = BotControlState.from_dict(payload)

        assert state.paused is True
        assert state.reason == "Test"
        assert state.updated_at is not None
        assert state.updated_at.year == 2024

    def test_from_dict_invalid_date(self) -> None:
        """Test from_dict with invalid date string."""
        payload = {
            "paused": False,
            "updated_at": "not-a-date",
        }

        state = BotControlState.from_dict(payload)

        assert state.paused is False
        assert state.updated_at is None

    def test_from_dict_non_string_reason(self) -> None:
        """Test from_dict with non-string reason."""
        payload = {
            "paused": True,
            "reason": 123,  # Should be ignored
        }

        state = BotControlState.from_dict(payload)

        assert state.reason is None

    def test_from_dict_empty(self) -> None:
        """Test from_dict with empty payload."""
        state = BotControlState.from_dict({})

        assert state.paused is False
        assert state.reason is None


class TestControlPath:
    """Tests for _control_path function."""

    def test_control_path(self, tmp_path: Path) -> None:
        """Test control path generation."""
        result = _control_path(tmp_path)

        assert result == tmp_path / "control.json"


class TestLoadBotControl:
    """Tests for load_bot_control function."""

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading when file doesn't exist."""
        state = load_bot_control(tmp_path)

        assert state.paused is False
        assert state.reason is None

    def test_load_valid_file(self, tmp_path: Path) -> None:
        """Test loading valid control file."""
        control_file = tmp_path / "control.json"
        control_file.write_text(
            json.dumps(
                {
                    "paused": True,
                    "reason": "Test pause",
                }
            )
        )

        state = load_bot_control(tmp_path)

        assert state.paused is True
        assert state.reason == "Test pause"

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading file with invalid JSON."""
        control_file = tmp_path / "control.json"
        control_file.write_text("not valid json {")

        state = load_bot_control(tmp_path)

        assert state.paused is False

    def test_load_non_dict_json(self, tmp_path: Path) -> None:
        """Test loading file with non-dict JSON."""
        control_file = tmp_path / "control.json"
        control_file.write_text('["item1", "item2"]')

        state = load_bot_control(tmp_path)

        assert state.paused is False


class TestSaveBotControl:
    """Tests for save_bot_control function."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test save creates control file."""
        state = BotControlState(paused=True, reason="Saving test")

        result = save_bot_control(tmp_path, state)

        assert (tmp_path / "control.json").exists()
        assert result.updated_at is not None

    def test_save_preserves_updated_at(self, tmp_path: Path) -> None:
        """Test save preserves existing updated_at."""
        now = datetime(2024, 1, 15, tzinfo=timezone.utc)
        state = BotControlState(paused=True, updated_at=now)

        result = save_bot_control(tmp_path, state)

        assert result.updated_at == now

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test save creates parent directories."""
        nested_dir = tmp_path / "nested" / "dir"
        state = BotControlState(paused=False)

        save_bot_control(nested_dir, state)

        assert (nested_dir / "control.json").exists()


class TestUpdateBotControl:
    """Tests for update_bot_control function."""

    def test_update_pause(self, tmp_path: Path) -> None:
        """Test updating to paused state."""
        result = update_bot_control(tmp_path, paused=True, reason="Manual stop")

        assert result.paused is True
        assert result.reason == "Manual stop"
        assert result.updated_at is not None

    def test_update_unpause(self, tmp_path: Path) -> None:
        """Test updating to unpaused state."""
        # First pause
        update_bot_control(tmp_path, paused=True)

        # Then unpause
        result = update_bot_control(tmp_path, paused=False)

        assert result.paused is False

    def test_update_persists_to_file(self, tmp_path: Path) -> None:
        """Test that update persists to file."""
        update_bot_control(tmp_path, paused=True, reason="Persist test")

        # Load and verify
        loaded = load_bot_control(tmp_path)

        assert loaded.paused is True
        assert loaded.reason == "Persist test"


class TestUtcNow:
    """Tests for _utcnow function."""

    def test_utcnow_returns_utc(self) -> None:
        """Test _utcnow returns UTC datetime."""
        result = _utcnow()

        assert result.tzinfo == timezone.utc

    def test_utcnow_is_recent(self) -> None:
        """Test _utcnow returns current time."""
        before = datetime.now(timezone.utc)
        result = _utcnow()
        after = datetime.now(timezone.utc)

        assert before <= result <= after
