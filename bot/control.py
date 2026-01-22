"""Helpers for coordinating manual pause/resume controls across bots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BotControlState:
    """Represents the manual pause/resume state and risk settings for a bot."""

    paused: bool = False
    reason: Optional[str] = None
    updated_at: Optional[datetime] = field(default=None)

    # Risk settings
    allow_shorting: bool = False
    allow_leverage: bool = False
    aggressive_mode: bool = False
    max_leverage: float = 1.0  # 1.0 = no leverage

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "paused": self.paused,
            "reason": self.reason,
            "allow_shorting": self.allow_shorting,
            "allow_leverage": self.allow_leverage,
            "aggressive_mode": self.aggressive_mode,
            "max_leverage": self.max_leverage,
        }
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BotControlState":
        updated_value = payload.get("updated_at")
        updated_at: Optional[datetime]
        if isinstance(updated_value, str):
            try:
                updated_at = datetime.fromisoformat(updated_value)
            except ValueError:
                updated_at = None
        else:
            updated_at = None

        return cls(
            paused=bool(payload.get("paused", False)),
            reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
            updated_at=updated_at,
            allow_shorting=bool(payload.get("allow_shorting", False)),
            allow_leverage=bool(payload.get("allow_leverage", False)),
            aggressive_mode=bool(payload.get("aggressive_mode", False)),
            max_leverage=float(payload.get("max_leverage", 1.0)),
        )


def _control_path(base_dir: Path) -> Path:
    return base_dir / "control.json"


def load_bot_control(base_dir: Path) -> BotControlState:
    """Return the current BotControlState if the control file exists."""

    path = _control_path(base_dir)
    if not path.exists():
        return BotControlState()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return BotControlState()
    if not isinstance(payload, dict):
        return BotControlState()
    return BotControlState.from_dict(payload)


def save_bot_control(base_dir: Path, control: BotControlState) -> BotControlState:
    """Persist the provided control state to disk."""

    path = _control_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if control.updated_at is None:
        control.updated_at = _utcnow()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(control.to_dict(), handle, indent=2)
    return control


def update_bot_control(
    base_dir: Path,
    *,
    paused: bool,
    reason: Optional[str] = None,
) -> BotControlState:
    """Create and persist a new BotControlState for the bot."""

    control = BotControlState(paused=paused, reason=reason, updated_at=_utcnow())
    return save_bot_control(base_dir, control)
