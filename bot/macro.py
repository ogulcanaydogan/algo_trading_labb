from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

IMPACT_WEIGHTS: Dict[str, float] = {
    "low": 0.8,
    "medium": 1.0,
    "high": 1.5,
    "critical": 2.0,
}

SENTIMENT_BIAS: Dict[str, float] = {
    "bullish": 0.6,
    "bearish": -0.6,
    "positive": 0.4,
    "negative": -0.4,
    "hawkish": -0.5,
    "dovish": 0.5,
    "risk-on": 0.35,
    "risk-off": -0.35,
    "neutral": 0.0,
}


@dataclass
class MacroEvent:
    """Structured representation of a macro or political event."""

    title: str
    category: str = "general"
    sentiment: str = "neutral"
    impact: str = "medium"
    bias: Optional[float] = None
    actor: Optional[str] = None
    assets: Dict[str, float] = field(default_factory=dict)
    interest_rate_expectation: Optional[str] = None
    summary: Optional[str] = None
    timestamp: Optional[str] = None
    source: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MacroEvent":
        data = dict(payload)
        data.setdefault("category", "general")
        data.setdefault("sentiment", "neutral")
        data.setdefault("impact", "medium")
        return cls(**data)

    def weight(self) -> float:
        return IMPACT_WEIGHTS.get(self.impact.lower(), 1.0)

    def derived_bias(self, symbol: str) -> float:
        if self.bias is not None:
            return max(-1.0, min(1.0, float(self.bias)))

        bias = SENTIMENT_BIAS.get(self.sentiment.lower(), 0.0)

        if self.actor and "trump" in self.actor.lower():
            bias += 0.2 if bias > 0 else -0.2

        asset_bias = 0.0
        if self.assets:
            asset_bias = float(self.assets.get(symbol, self.assets.get("*", 0.0)))

        return max(-1.0, min(1.0, bias + asset_bias))

    def as_dict(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "title": self.title,
            "category": self.category,
            "sentiment": self.sentiment,
            "impact": self.impact,
            "actor": self.actor,
            "interest_rate_expectation": self.interest_rate_expectation,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "source": self.source,
            "bias": self.bias,
        }
        if symbol:
            try:
                payload["derived_bias"] = round(self.derived_bias(symbol), 4)
            except Exception:
                payload["derived_bias"] = None
        return payload


@dataclass
class MacroInsight:
    """Aggregated macro sentiment for a target symbol."""

    symbol: str
    bias_score: float
    confidence: float
    summary: str
    drivers: List[str] = field(default_factory=list)
    interest_rate_outlook: Optional[str] = None
    political_risk: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def neutral(cls, symbol: str) -> "MacroInsight":
        return cls(
            symbol=symbol,
            bias_score=0.0,
            confidence=0.0,
            summary="No macro catalysts detected for the configured symbol.",
            drivers=[],
            interest_rate_outlook=None,
            political_risk=None,
            events=[],
        )


class MacroSentimentEngine:
    """Fuse macro and geopolitical events into a trading bias."""

    def __init__(
        self,
        events_path: Optional[Path] = None,
        refresh_interval: int = 300,
        baseline_events: Optional[Iterable[MacroEvent]] = None,
    ) -> None:
        self.events_path = events_path
        self.refresh_interval = max(30, refresh_interval)
        self._baseline_events = list(baseline_events or self._build_default_events())
        self._events: List[MacroEvent] = list(self._baseline_events)
        self._last_loaded: float = 0.0
        self._dynamic_templates: List[Dict[str, Any]] = self._build_dynamic_templates()
        self._cycle_index: int = 0

    @classmethod
    def from_env(cls) -> "MacroSentimentEngine":
        path_value = os.getenv("MACRO_EVENTS_PATH")
        refresh = int(os.getenv("MACRO_REFRESH_SECONDS", "300"))
        events_path = Path(path_value).expanduser() if path_value else None
        return cls(events_path=events_path, refresh_interval=refresh)

    def refresh_if_needed(self) -> None:
        now = time.time()
        if self.events_path is None:
            self._events = self._compose_events()
            return
        if now - self._last_loaded < self.refresh_interval:
            return
        try:
            self._events = self._compose_events(
                extra_events=self._load_events_from_path(self.events_path)
            )
            self._last_loaded = now
        except (OSError, json.JSONDecodeError):
            self._events = self._compose_events()

    def assess(self, symbol: str) -> MacroInsight:
        self.refresh_if_needed()
        relevant = self._select_events(symbol)
        if not relevant:
            return MacroInsight.neutral(symbol)

        weighted_bias = 0.0
        total_weight = 0.0
        political_notes: List[str] = []
        rate_notes: List[str] = []

        for event in relevant:
            weight = event.weight()
            timestamp_value = event.timestamp
            if timestamp_value:
                try:
                    event_time = datetime.fromisoformat(
                        timestamp_value.replace("Z", "+00:00")
                    )
                    age_hours = (
                        datetime.now(timezone.utc) - event_time
                    ).total_seconds() / 3600
                    decay = math.exp(-max(age_hours, 0.0) / 6.0)
                    weight *= max(0.35, decay)
                except ValueError:
                    pass
            bias = event.derived_bias(symbol)
            weighted_bias += bias * weight
            total_weight += weight

            if event.category.lower() in {"politics", "geopolitics"}:
                descriptor = event.summary or event.title
                if event.actor:
                    descriptor = f"{event.actor}: {descriptor}"
                political_notes.append(descriptor)
            if event.category.lower() in {"central_bank", "fed", "rates"}:
                rate_hint = event.interest_rate_expectation or event.summary or event.title
                rate_notes.append(rate_hint)

        bias_score = weighted_bias / total_weight if total_weight else 0.0
        confidence = min(1.0, total_weight / (len(relevant) * 1.6)) if relevant else 0.0

        sorted_events = sorted(
            relevant,
            key=lambda item: abs(item.derived_bias(symbol)) * item.weight(),
            reverse=True,
        )

        drivers = [
            f"{event.title} ({event.sentiment}, {event.impact} impact)"
            for event in sorted_events[:4]
        ]
        events_payload = [event.as_dict(symbol) for event in sorted_events[:6]]

        bias_text = "bullish" if bias_score > 0.05 else "bearish" if bias_score < -0.05 else "balanced"
        summary = (
            f"Macro bias is {bias_text} ({bias_score:+.2f}) based on {len(relevant)} tracked catalysts."
        )

        if political_notes:
            political_section = "; ".join(political_notes[:2])
        else:
            political_section = None

        interest_section = rate_notes[0] if rate_notes else None

        if interest_section:
            summary += f" Fed watch: {interest_section}."
        if political_section:
            summary += f" Political risk: {political_section}."

        return MacroInsight(
            symbol=symbol,
            bias_score=round(bias_score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            drivers=drivers,
            interest_rate_outlook=interest_section,
            political_risk=political_section,
            events=events_payload,
        )

    def _select_events(self, symbol: str) -> List[MacroEvent]:
        selected: List[MacroEvent] = []
        for event in self._events:
            if not event.assets:
                selected.append(event)
                continue
            keys = {key.lower() for key in event.assets.keys()}
            if "*" in keys or symbol.lower() in keys:
                selected.append(event)
        return selected[:12]

    def _load_events_from_path(self, path: Path) -> List[MacroEvent]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        events: List[MacroEvent] = []
        if isinstance(payload, list):
            for entry in payload:
                try:
                    events.append(MacroEvent.from_dict(entry))
                except TypeError:
                    continue
        return events

    def _compose_events(
        self,
        extra_events: Optional[Iterable[MacroEvent]] = None,
    ) -> List[MacroEvent]:
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline = self._refresh_baseline_events(now)
        dynamic = self._generate_dynamic_events(now)
        combined: List[MacroEvent] = baseline + dynamic
        if extra_events:
            combined.extend(extra_events)
        # ensure newest events first
        combined.sort(key=lambda event: event.timestamp or now, reverse=True)
        return combined[:20]

    def _generate_dynamic_events(self, timestamp: str) -> List[MacroEvent]:
        events: List[MacroEvent] = []
        if not self._dynamic_templates:
            return events
        templates_to_use = min(5, len(self._dynamic_templates))
        for offset in range(templates_to_use):
            template = self._dynamic_templates[
                (self._cycle_index + offset) % len(self._dynamic_templates)
            ]
            payload = dict(template)
            bias_jitter = math.sin((self._cycle_index + offset) / 2.5) * 0.1
            payload["timestamp"] = timestamp
            update_tag = timestamp.split("T")[-1][:5] if "T" in timestamp else timestamp
            payload["summary"] = f"{template['summary']} (refresh {update_tag}Z)"
            if "bias" in payload and payload["bias"] is not None:
                payload["bias"] = max(
                    -1.0,
                    min(1.0, float(payload["bias"]) + bias_jitter),
                )
            events.append(MacroEvent.from_dict(payload))
        self._cycle_index = (self._cycle_index + 1) % len(self._dynamic_templates)
        return events

    def _refresh_baseline_events(self, timestamp: str) -> List[MacroEvent]:
        if not self._baseline_events:
            return []

        ordered = (
            self._baseline_events[self._cycle_index % len(self._baseline_events) :]
            + self._baseline_events[: self._cycle_index % len(self._baseline_events)]
        )

        refreshed: List[MacroEvent] = []
        for index, event in enumerate(ordered):
            payload = asdict(event)
            payload["timestamp"] = timestamp
            base_summary = event.summary or event.title
            update_tag = timestamp.split("T")[-1][:5] if "T" in timestamp else timestamp
            payload["summary"] = f"{base_summary} (update {update_tag}Z)"
            bias_value = payload.get("bias")
            if bias_value is not None:
                jitter = math.sin((self._cycle_index + index) / 3.0) * 0.08
                payload["bias"] = max(-1.0, min(1.0, float(bias_value) + jitter))
            refreshed.append(MacroEvent.from_dict(payload))
        return refreshed

    def _build_default_events(self) -> List[MacroEvent]:
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return [
            MacroEvent(
                title="Trump vows fresh tariffs review",
                category="politics",
                sentiment="bearish",
                impact="high",
                actor="Donald Trump",
                summary="Potential tariff escalation keeps risk assets cautious.",
                timestamp=now,
                assets={"BTC/USDT": -0.15, "ETH/USDT": -0.1, "*": -0.05},
            ),
            MacroEvent(
                title="Fed officials guide for data-dependent path",
                category="central_bank",
                sentiment="neutral",
                impact="medium",
                interest_rate_expectation="Fed likely to keep rates unchanged but watch core inflation prints.",
                timestamp=now,
            ),
            MacroEvent(
                title="US payrolls surprise to upside",
                category="macro",
                sentiment="hawkish",
                impact="high",
                summary="Stronger labor data tempers rate-cut odds.",
                timestamp=now,
            ),
        ]

    def _build_dynamic_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Global supply chains tighten",
                "category": "macro",
                "sentiment": "negative",
                "impact": "high",
                "summary": "Freight rates spike as Red Sea tensions disrupt shipping lanes",
                "assets": {"XAU/USD": 0.18, "XAG/USD": 0.12, "USOIL/USD": 0.15, "*": -0.05},
            },
            {
                "title": "Fed speakers lean data-dependent",
                "category": "central_bank",
                "sentiment": "neutral",
                "impact": "medium",
                "interest_rate_expectation": "Short-term rates stay restrictive until inflation cools further.",
                "summary": "Officials reiterate patience on cuts while monitoring labour data",
            },
            {
                "title": "Fiscal policy debate heats up",
                "category": "politics",
                "sentiment": "bearish",
                "impact": "high",
                "actor": "US Congress",
                "summary": "Budget standoff revives volatility across risk assets",
                "assets": {"BTC/USDT": -0.1, "ETH/USDT": -0.08, "*": -0.04},
            },
            {
                "title": "Energy markets eye OPEC guidance",
                "category": "commodities",
                "sentiment": "bullish",
                "impact": "medium",
                "summary": "Production discipline hints at tighter crude balances into year-end",
                "assets": {"USOIL/USD": 0.22, "XAU/USD": 0.05},
            },
            {
                "title": "Tech earnings beat expectations",
                "category": "equities",
                "sentiment": "positive",
                "impact": "medium",
                "summary": "Mega-cap guidance lifts sentiment for growth equities",
                "assets": {"AAPL": 0.1, "NVDA": 0.12, "*": 0.04},
            },
        ]

