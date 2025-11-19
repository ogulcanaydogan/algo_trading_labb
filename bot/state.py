from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

PositionType = Literal["LONG", "SHORT", "FLAT"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BotState:
    """Represents the latest trading state for a single instrument."""

    timestamp: datetime = field(default_factory=_utcnow)
    symbol: str = "BTC/USDT"
    position: PositionType = "FLAT"
    entry_price: Optional[float] = None
    position_size: float = 0.0
    balance: float = 10_000.0
    initial_balance: float = 10_000.0
    unrealized_pnl_pct: float = 0.0
    last_signal: Optional[str] = None
    last_signal_reason: Optional[str] = None
    confidence: Optional[float] = None
    technical_signal: Optional[str] = None
    technical_confidence: Optional[float] = None
    technical_reason: Optional[str] = None
    ai_override_active: bool = False
    rsi: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    risk_per_trade_pct: float = 0.5
    ai_action: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_probability_long: Optional[float] = None
    ai_probability_short: Optional[float] = None
    ai_probability_flat: Optional[float] = None
    ai_expected_move_pct: Optional[float] = None
    ai_summary: Optional[str] = None
    ai_features: Dict[str, float] = field(default_factory=dict)
    macro_bias: Optional[float] = None
    macro_confidence: Optional[float] = None
    macro_summary: Optional[str] = None
    macro_drivers: List[str] = field(default_factory=list)
    macro_interest_rate_outlook: Optional[str] = None
    macro_political_risk: Optional[str] = None
    macro_events: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_playbook: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotState":
        payload = dict(data)
        ts = payload.get("timestamp")
        if ts:
            payload["timestamp"] = datetime.fromisoformat(ts)
        else:
            payload["timestamp"] = _utcnow()
        return cls(**payload)


@dataclass
class SignalEvent:
    timestamp: datetime
    symbol: str
    decision: PositionType
    confidence: float
    reason: str
    ai_action: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_expected_move_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload.setdefault("execution_reason", self.reason)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEvent":
        payload = dict(data)
        payload["timestamp"] = datetime.fromisoformat(payload["timestamp"])
        if "execution_reason" in payload and "reason" not in payload:
            payload["reason"] = payload["execution_reason"]
        return cls(**payload)


@dataclass
class EquityPoint:
    timestamp: datetime
    value: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquityPoint":
        payload = dict(data)
        payload["timestamp"] = datetime.fromisoformat(payload["timestamp"])
        return cls(**payload)


class StateStore:
    """Simple JSON-backed store used by the bot and the status API."""

    def __init__(
        self,
        state_path: Path,
        signals_path: Path,
        equity_path: Path,
        max_signals: int = 250,
    ) -> None:
        self.state_path = state_path
        self.signals_path = signals_path
        self.equity_path = equity_path
        self.max_signals = max_signals
        self._lock = threading.Lock()

        self.state: BotState = BotState()
        self.signals: List[SignalEvent] = []
        self.equity_curve: List[EquityPoint] = []

        self._ensure_parent_dirs()
        self.load()

    def _ensure_parent_dirs(self) -> None:
        for path in (self.state_path, self.signals_path, self.equity_path):
            path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        with self._lock:
            if self.state_path.exists():
                self.state = BotState.from_dict(self._read_json(self.state_path))
            if self.signals_path.exists():
                items = self._read_json(self.signals_path)
                self.signals = [SignalEvent.from_dict(item) for item in items]
            if self.equity_path.exists():
                points = self._read_json(self.equity_path)
                self.equity_curve = [EquityPoint.from_dict(item) for item in points]

    def update_state(self, **kwargs: Any) -> BotState:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            self.state.timestamp = _utcnow()
            self._flush_state()
            return self.state

    def record_signal(self, signal: SignalEvent) -> None:
        with self._lock:
            self.signals.append(signal)
            self.signals = self.signals[-self.max_signals :]
            self._write_json(
                self.signals_path,
                [entry.to_dict() for entry in self.signals],
            )

    def record_equity(self, equity_point: EquityPoint) -> None:
        with self._lock:
            self.equity_curve.append(equity_point)
            self.equity_curve = self.equity_curve[-10_000:]
            self._write_json(
                self.equity_path,
                [point.to_dict() for point in self.equity_curve],
            )

    def get_state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return self.state.to_dict()

    def get_signals(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            collection = self.signals[-limit:] if limit else self.signals
            return [signal.to_dict() for signal in reversed(collection)]

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [point.to_dict() for point in self.equity_curve]

    def _flush_state(self) -> None:
        self._write_json(self.state_path, self.state.to_dict())

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)


def create_state_store(base_dir: Path) -> StateStore:
    state_path = base_dir / "state.json"
    signals_path = base_dir / "signals.json"
    equity_path = base_dir / "equity.json"
    return StateStore(state_path, signals_path, equity_path)


def load_bot_state_from_path(path: Path) -> Optional[BotState]:
    """Load a BotState from the given JSON file if it exists."""

    if not path.exists() or not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return BotState.from_dict(payload)
