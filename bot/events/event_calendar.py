"""
Event Calendar.

Tracks and manages market events:
- Economic events (FOMC, CPI, NFP)
- Crypto events (halvings, upgrades)
- Pre-event actions (reduce size)
- Post-event analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """Event impact level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Type of market event."""
    ECONOMIC = "economic"
    CRYPTO = "crypto"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"  # e.g., options expiry


@dataclass
class MarketEvent:
    """A market event."""
    id: str
    name: str
    event_type: EventType
    impact: EventImpact
    scheduled_time: datetime
    symbols_affected: List[str]
    description: str = ""
    expected_volatility_increase: float = 0.0  # Expected % increase
    pre_event_action: str = "reduce_size"  # reduce_size, hedge, close, none
    pre_event_hours: int = 24  # Hours before event to take action

    @property
    def is_upcoming(self) -> bool:
        return self.scheduled_time > datetime.now()

    @property
    def hours_until(self) -> float:
        delta = self.scheduled_time - datetime.now()
        return delta.total_seconds() / 3600

    @property
    def is_in_pre_event_window(self) -> bool:
        return 0 < self.hours_until <= self.pre_event_hours


@dataclass
class EventAction:
    """Recommended action for an event."""
    event: MarketEvent
    action: str  # reduce_size, hedge, close, alert
    symbols: List[str]
    size_multiplier: float  # e.g., 0.5 = reduce to 50%
    reason: str


class EventCalendar:
    """
    Manages market event calendar.

    Features:
    - Event scheduling
    - Pre-event alerts
    - Automatic position adjustments
    - Historical event analysis
    """

    # Recurring economic events (day of week, approximate time UTC)
    RECURRING_EVENTS = {
        "FOMC": {
            "impact": EventImpact.CRITICAL,
            "pre_event_hours": 24,
            "volatility_increase": 0.5,
            "symbols": ["BTC/USDT", "ETH/USDT", "SPX500"],
        },
        "CPI": {
            "impact": EventImpact.HIGH,
            "pre_event_hours": 12,
            "volatility_increase": 0.3,
            "symbols": ["BTC/USDT", "ETH/USDT"],
        },
        "NFP": {
            "impact": EventImpact.HIGH,
            "pre_event_hours": 12,
            "volatility_increase": 0.25,
            "symbols": ["SPX500", "EUR/USD"],
        },
        "PCE": {
            "impact": EventImpact.MEDIUM,
            "pre_event_hours": 6,
            "volatility_increase": 0.2,
            "symbols": ["BTC/USDT", "SPX500"],
        },
    }

    def __init__(
        self,
        event_callback: Optional[Callable[[EventAction], None]] = None,
    ):
        self.event_callback = event_callback

        # Scheduled events
        self._events: Dict[str, MarketEvent] = {}

        # Past events
        self._past_events: List[MarketEvent] = []

        # Alerts sent (to avoid duplicates)
        self._alerts_sent: Dict[str, datetime] = {}

        logger.info("EventCalendar initialized")

    def add_event(self, event: MarketEvent):
        """Add an event to the calendar."""
        self._events[event.id] = event
        logger.info(f"Added event: {event.name} at {event.scheduled_time}")

    def add_economic_event(
        self,
        name: str,
        scheduled_time: datetime,
        custom_impact: Optional[EventImpact] = None,
    ):
        """Add a known economic event."""
        if name in self.RECURRING_EVENTS:
            config = self.RECURRING_EVENTS[name]
            event = MarketEvent(
                id=f"{name}_{scheduled_time.isoformat()}",
                name=name,
                event_type=EventType.ECONOMIC,
                impact=custom_impact or config["impact"],
                scheduled_time=scheduled_time,
                symbols_affected=config["symbols"],
                expected_volatility_increase=config["volatility_increase"],
                pre_event_hours=config["pre_event_hours"],
            )
            self.add_event(event)
        else:
            logger.warning(f"Unknown economic event: {name}")

    def add_crypto_event(
        self,
        name: str,
        scheduled_time: datetime,
        symbols: List[str],
        impact: EventImpact = EventImpact.HIGH,
        description: str = "",
    ):
        """Add a crypto-specific event."""
        event = MarketEvent(
            id=f"crypto_{name}_{scheduled_time.isoformat()}",
            name=name,
            event_type=EventType.CRYPTO,
            impact=impact,
            scheduled_time=scheduled_time,
            symbols_affected=symbols,
            description=description,
            pre_event_hours=48 if impact == EventImpact.CRITICAL else 24,
        )
        self.add_event(event)

    def check_upcoming_events(self) -> List[EventAction]:
        """
        Check for upcoming events and return actions.

        Returns actions for events in their pre-event window.
        """
        actions = []
        now = datetime.now()

        for event_id, event in list(self._events.items()):
            # Move past events to history
            if not event.is_upcoming:
                self._past_events.append(event)
                del self._events[event_id]
                continue

            # Check if in pre-event window
            if event.is_in_pre_event_window:
                # Check if alert already sent
                alert_key = f"{event_id}_{event.pre_event_action}"
                if alert_key in self._alerts_sent:
                    last_alert = self._alerts_sent[alert_key]
                    if (now - last_alert).total_seconds() < 3600:  # 1 hour cooldown
                        continue

                action = self._generate_event_action(event)
                if action:
                    actions.append(action)
                    self._alerts_sent[alert_key] = now

                    # Trigger callback
                    if self.event_callback:
                        try:
                            self.event_callback(action)
                        except Exception as e:
                            logger.error(f"Event callback failed: {e}")

        return actions

    def _generate_event_action(self, event: MarketEvent) -> Optional[EventAction]:
        """Generate action for an event."""
        # Determine action based on impact
        if event.impact == EventImpact.CRITICAL:
            action = "reduce_size"
            multiplier = 0.3  # Reduce to 30%
        elif event.impact == EventImpact.HIGH:
            action = "reduce_size"
            multiplier = 0.5  # Reduce to 50%
        elif event.impact == EventImpact.MEDIUM:
            action = "alert"
            multiplier = 0.7
        else:
            action = "alert"
            multiplier = 1.0

        return EventAction(
            event=event,
            action=action,
            symbols=event.symbols_affected,
            size_multiplier=multiplier,
            reason=f"{event.name} in {event.hours_until:.1f}h - Impact: {event.impact.value}",
        )

    def get_events_for_symbol(self, symbol: str, hours: int = 48) -> List[MarketEvent]:
        """Get upcoming events affecting a symbol."""
        cutoff = datetime.now() + timedelta(hours=hours)
        return [
            e for e in self._events.values()
            if symbol in e.symbols_affected and e.scheduled_time < cutoff
        ]

    def get_next_critical_event(self) -> Optional[MarketEvent]:
        """Get the next critical/high impact event."""
        critical_events = [
            e for e in self._events.values()
            if e.impact in [EventImpact.CRITICAL, EventImpact.HIGH] and e.is_upcoming
        ]

        if critical_events:
            return min(critical_events, key=lambda e: e.scheduled_time)
        return None

    def should_reduce_position(self, symbol: str) -> Tuple[bool, float, str]:
        """
        Check if positions should be reduced for a symbol.

        Returns:
            Tuple of (should_reduce, multiplier, reason)
        """
        events = self.get_events_for_symbol(symbol, hours=24)

        if not events:
            return False, 1.0, ""

        # Find highest impact event
        highest_impact = max(events, key=lambda e: list(EventImpact).index(e.impact))

        if highest_impact.is_in_pre_event_window:
            if highest_impact.impact == EventImpact.CRITICAL:
                return True, 0.3, f"Critical event: {highest_impact.name}"
            elif highest_impact.impact == EventImpact.HIGH:
                return True, 0.5, f"High impact event: {highest_impact.name}"

        return False, 1.0, ""

    def get_upcoming_events(self, hours: int = 168) -> List[MarketEvent]:
        """Get all upcoming events in next N hours."""
        cutoff = datetime.now() + timedelta(hours=hours)
        events = [e for e in self._events.values() if e.scheduled_time < cutoff]
        return sorted(events, key=lambda e: e.scheduled_time)

    def get_stats(self) -> Dict:
        """Get calendar statistics."""
        upcoming = self.get_upcoming_events(168)
        return {
            "total_scheduled": len(self._events),
            "upcoming_24h": len([e for e in upcoming if e.hours_until <= 24]),
            "upcoming_week": len(upcoming),
            "critical_upcoming": len([e for e in upcoming if e.impact == EventImpact.CRITICAL]),
            "past_events": len(self._past_events),
            "next_event": upcoming[0].name if upcoming else None,
            "next_event_hours": upcoming[0].hours_until if upcoming else None,
        }


# Singleton
_event_calendar: Optional[EventCalendar] = None


def get_event_calendar(
    event_callback: Optional[Callable[[EventAction], None]] = None
) -> EventCalendar:
    """Get or create the EventCalendar singleton."""
    global _event_calendar
    if _event_calendar is None:
        _event_calendar = EventCalendar(event_callback=event_callback)
    return _event_calendar
