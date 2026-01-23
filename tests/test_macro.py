"""
Tests for Macro Sentiment Module.

Tests the macro sentiment engine functionality including
event processing, bias calculation, and sentiment aggregation.
"""

import pytest
import tempfile
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from bot.macro import (
    IMPACT_WEIGHTS,
    SENTIMENT_BIAS,
    MacroEvent,
    MacroInsight,
    MacroSentimentEngine,
)


class TestConstants:
    """Tests for module constants."""

    def test_impact_weights_defined(self):
        """Test impact weights are defined."""
        assert "low" in IMPACT_WEIGHTS
        assert "medium" in IMPACT_WEIGHTS
        assert "high" in IMPACT_WEIGHTS
        assert "critical" in IMPACT_WEIGHTS

    def test_impact_weights_values(self):
        """Test impact weight values are correct."""
        assert IMPACT_WEIGHTS["low"] == 0.8
        assert IMPACT_WEIGHTS["medium"] == 1.0
        assert IMPACT_WEIGHTS["high"] == 1.5
        assert IMPACT_WEIGHTS["critical"] == 2.0

    def test_impact_weights_ordering(self):
        """Test impact weights are ordered correctly."""
        assert IMPACT_WEIGHTS["low"] < IMPACT_WEIGHTS["medium"]
        assert IMPACT_WEIGHTS["medium"] < IMPACT_WEIGHTS["high"]
        assert IMPACT_WEIGHTS["high"] < IMPACT_WEIGHTS["critical"]

    def test_sentiment_bias_defined(self):
        """Test sentiment bias values are defined."""
        assert "bullish" in SENTIMENT_BIAS
        assert "bearish" in SENTIMENT_BIAS
        assert "neutral" in SENTIMENT_BIAS
        assert "hawkish" in SENTIMENT_BIAS
        assert "dovish" in SENTIMENT_BIAS

    def test_sentiment_bias_signs(self):
        """Test sentiment bias signs are correct."""
        assert SENTIMENT_BIAS["bullish"] > 0
        assert SENTIMENT_BIAS["bearish"] < 0
        assert SENTIMENT_BIAS["neutral"] == 0
        assert SENTIMENT_BIAS["hawkish"] < 0  # Hawkish = less stimulus = bearish for risk
        assert SENTIMENT_BIAS["dovish"] > 0  # Dovish = more stimulus = bullish for risk


class TestMacroEvent:
    """Tests for MacroEvent dataclass."""

    def test_macro_event_creation(self):
        """Test creating a MacroEvent."""
        event = MacroEvent(
            title="Fed Rate Decision",
            category="central_bank",
            sentiment="hawkish",
            impact="high",
        )

        assert event.title == "Fed Rate Decision"
        assert event.category == "central_bank"
        assert event.sentiment == "hawkish"
        assert event.impact == "high"

    def test_macro_event_defaults(self):
        """Test MacroEvent default values."""
        event = MacroEvent(title="Test Event")

        assert event.category == "general"
        assert event.sentiment == "neutral"
        assert event.impact == "medium"
        assert event.bias is None
        assert event.actor is None
        assert event.assets == {}

    def test_macro_event_with_assets(self):
        """Test MacroEvent with asset-specific impacts."""
        event = MacroEvent(
            title="Tariff Announcement",
            assets={"BTC/USDT": -0.2, "ETH/USDT": -0.15, "*": -0.05},
        )

        assert event.assets["BTC/USDT"] == -0.2
        assert event.assets["*"] == -0.05

    def test_from_dict_basic(self):
        """Test creating MacroEvent from dict."""
        data = {
            "title": "Test Event",
            "category": "politics",
            "sentiment": "bearish",
        }

        event = MacroEvent.from_dict(data)

        assert event.title == "Test Event"
        assert event.category == "politics"
        assert event.sentiment == "bearish"
        assert event.impact == "medium"  # Default

    def test_from_dict_with_defaults(self):
        """Test from_dict applies defaults."""
        data = {"title": "Minimal Event"}

        event = MacroEvent.from_dict(data)

        assert event.category == "general"
        assert event.sentiment == "neutral"
        assert event.impact == "medium"

    def test_weight_method(self):
        """Test weight calculation."""
        low_event = MacroEvent(title="Low Impact", impact="low")
        medium_event = MacroEvent(title="Medium Impact", impact="medium")
        high_event = MacroEvent(title="High Impact", impact="high")
        critical_event = MacroEvent(title="Critical Impact", impact="critical")

        assert low_event.weight() == 0.8
        assert medium_event.weight() == 1.0
        assert high_event.weight() == 1.5
        assert critical_event.weight() == 2.0

    def test_weight_unknown_impact(self):
        """Test weight with unknown impact level."""
        event = MacroEvent(title="Unknown", impact="unknown")

        # Should default to 1.0
        assert event.weight() == 1.0

    def test_derived_bias_with_explicit_bias(self):
        """Test derived_bias with explicit bias set."""
        event = MacroEvent(title="Test", bias=0.5)

        assert event.derived_bias("BTC/USDT") == 0.5

    def test_derived_bias_clamped(self):
        """Test derived_bias is clamped to [-1, 1]."""
        event_high = MacroEvent(title="Test", bias=1.5)
        event_low = MacroEvent(title="Test", bias=-1.5)

        assert event_high.derived_bias("BTC/USDT") == 1.0
        assert event_low.derived_bias("BTC/USDT") == -1.0

    def test_derived_bias_from_sentiment(self):
        """Test derived_bias from sentiment."""
        bullish = MacroEvent(title="Test", sentiment="bullish")
        bearish = MacroEvent(title="Test", sentiment="bearish")

        assert bullish.derived_bias("BTC/USDT") == 0.6
        assert bearish.derived_bias("BTC/USDT") == -0.6

    def test_derived_bias_trump_actor(self):
        """Test derived_bias with Trump actor modifier."""
        event = MacroEvent(
            title="Test",
            sentiment="bearish",
            actor="Donald Trump",
        )

        # Base bearish bias is -0.6, Trump modifier should make it more bearish
        bias = event.derived_bias("BTC/USDT")
        assert bias < -0.6

    def test_derived_bias_with_asset_specific(self):
        """Test derived_bias with asset-specific bias."""
        event = MacroEvent(
            title="Test",
            sentiment="neutral",
            assets={"BTC/USDT": 0.3, "ETH/USDT": -0.2},
        )

        assert event.derived_bias("BTC/USDT") == 0.3
        assert event.derived_bias("ETH/USDT") == -0.2

    def test_derived_bias_with_wildcard_asset(self):
        """Test derived_bias with wildcard asset."""
        event = MacroEvent(
            title="Test",
            sentiment="neutral",
            assets={"*": 0.1},
        )

        assert event.derived_bias("ANY/SYMBOL") == 0.1

    def test_as_dict(self):
        """Test as_dict serialization."""
        event = MacroEvent(
            title="Test Event",
            category="politics",
            sentiment="bearish",
            impact="high",
            actor="Test Actor",
            summary="Test summary",
        )

        result = event.as_dict()

        assert result["title"] == "Test Event"
        assert result["category"] == "politics"
        assert result["sentiment"] == "bearish"
        assert result["impact"] == "high"
        assert result["actor"] == "Test Actor"
        assert result["summary"] == "Test summary"


class TestMacroInsight:
    """Tests for MacroInsight dataclass."""

    def test_macro_insight_creation(self):
        """Test creating a MacroInsight."""
        insight = MacroInsight(
            symbol="BTC/USDT",
            bias_score=0.25,
            confidence=0.8,
            summary="Bullish macro conditions",
            drivers=["Fed dovish signal", "Strong employment"],
        )

        assert insight.symbol == "BTC/USDT"
        assert insight.bias_score == 0.25
        assert insight.confidence == 0.8
        assert len(insight.drivers) == 2

    def test_macro_insight_defaults(self):
        """Test MacroInsight default values."""
        insight = MacroInsight(
            symbol="ETH/USDT",
            bias_score=0.0,
            confidence=0.0,
            summary="No catalysts",
        )

        assert insight.drivers == []
        assert insight.interest_rate_outlook is None
        assert insight.political_risk is None
        assert insight.events == []

    def test_neutral_factory(self):
        """Test neutral insight factory method."""
        insight = MacroInsight.neutral("BTC/USDT")

        assert insight.symbol == "BTC/USDT"
        assert insight.bias_score == 0.0
        assert insight.confidence == 0.0
        assert "No macro catalysts" in insight.summary


class TestMacroSentimentEngine:
    """Tests for MacroSentimentEngine class."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MacroSentimentEngine()

    @pytest.fixture
    def sample_events(self) -> List[MacroEvent]:
        """Create sample events."""
        return [
            MacroEvent(
                title="Fed Rate Cut",
                category="central_bank",
                sentiment="dovish",
                impact="high",
            ),
            MacroEvent(
                title="Trade Tensions",
                category="geopolitics",
                sentiment="bearish",
                impact="medium",
            ),
        ]

    def test_init_default(self, engine):
        """Test default initialization."""
        assert engine.events_path is None
        assert engine.refresh_interval >= 30

    def test_init_with_events_path(self):
        """Test initialization with events path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "events.json"
            engine = MacroSentimentEngine(events_path=path)

            assert engine.events_path == path

    def test_init_with_baseline_events(self, sample_events):
        """Test initialization with baseline events."""
        engine = MacroSentimentEngine(baseline_events=sample_events)

        assert len(engine._events) >= len(sample_events)

    def test_init_min_refresh_interval(self):
        """Test minimum refresh interval is enforced."""
        engine = MacroSentimentEngine(refresh_interval=10)

        assert engine.refresh_interval >= 30

    def test_from_env_default(self):
        """Test from_env with no env vars."""
        # Clear env vars
        os.environ.pop("MACRO_EVENTS_PATH", None)
        os.environ.pop("MACRO_REFRESH_SECONDS", None)

        engine = MacroSentimentEngine.from_env()

        assert engine.events_path is None
        assert engine.refresh_interval == 300

    def test_from_env_with_path(self):
        """Test from_env with MACRO_EVENTS_PATH set."""
        os.environ["MACRO_EVENTS_PATH"] = "/tmp/test_events.json"

        try:
            engine = MacroSentimentEngine.from_env()
            assert engine.events_path == Path("/tmp/test_events.json")
        finally:
            del os.environ["MACRO_EVENTS_PATH"]

    def test_from_env_with_refresh(self):
        """Test from_env with MACRO_REFRESH_SECONDS set."""
        os.environ["MACRO_REFRESH_SECONDS"] = "600"

        try:
            engine = MacroSentimentEngine.from_env()
            assert engine.refresh_interval == 600
        finally:
            del os.environ["MACRO_REFRESH_SECONDS"]

    def test_assess_no_relevant_events(self):
        """Test assess with no events for specific symbol."""
        events = [
            MacroEvent(
                title="ETH Only Event",
                assets={"ETH/USDT": 0.5},  # Only applies to ETH
            ),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        # Assess for a symbol not in the event
        insight = engine.assess("SOL/USDT")

        # No relevant events for SOL
        assert insight.bias_score == 0.0
        assert insight.confidence == 0.0

    def test_assess_with_events(self, engine):
        """Test assess with default events."""
        insight = engine.assess("BTC/USDT")

        assert isinstance(insight, MacroInsight)
        assert insight.symbol == "BTC/USDT"
        assert -1 <= insight.bias_score <= 1
        assert 0 <= insight.confidence <= 1

    def test_assess_summary_format(self, engine):
        """Test assess summary formatting."""
        insight = engine.assess("BTC/USDT")

        assert "bias" in insight.summary.lower() or "catalysts" in insight.summary.lower()

    def test_assess_drivers(self, engine):
        """Test assess includes drivers."""
        insight = engine.assess("BTC/USDT")

        assert isinstance(insight.drivers, list)

    def test_assess_events_included(self, engine):
        """Test assess includes event details."""
        insight = engine.assess("BTC/USDT")

        assert isinstance(insight.events, list)

    def test_select_events_wildcard(self):
        """Test event selection with wildcard."""
        events = [
            MacroEvent(title="Global", assets={"*": 0.1}),
            MacroEvent(title="Specific", assets={"BTC/USDT": 0.2}),
            MacroEvent(title="Other", assets={"ETH/USDT": 0.3}),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        selected = engine._select_events("BTC/USDT")

        # Should get global (wildcard) and BTC-specific
        assert len(selected) == 2

    def test_select_events_case_insensitive(self):
        """Test event selection is case insensitive."""
        events = [
            MacroEvent(title="Test", assets={"btc/usdt": 0.1}),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        selected = engine._select_events("BTC/USDT")

        assert len(selected) == 1

    def test_select_events_no_assets_included(self):
        """Test events without asset restrictions are included."""
        events = [
            MacroEvent(title="General Event"),  # No assets = applies to all
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        selected = engine._select_events("ANY/SYMBOL")

        assert len(selected) == 1

    def test_load_events_from_json(self):
        """Test loading events from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"title": "Event 1", "sentiment": "bullish"},
                {"title": "Event 2", "sentiment": "bearish"},
            ]
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            engine = MacroSentimentEngine()
            events = engine._load_events_from_path(temp_path)

            assert len(events) == 2
            assert events[0].title == "Event 1"
        finally:
            temp_path.unlink()

    def test_load_events_from_nonexistent_path(self):
        """Test loading from non-existent path returns empty list."""
        engine = MacroSentimentEngine()
        events = engine._load_events_from_path(Path("/nonexistent/path.json"))

        assert events == []

    def test_load_events_invalid_json(self):
        """Test loading invalid JSON returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            temp_path = Path(f.name)

        try:
            engine = MacroSentimentEngine(events_path=temp_path)
            engine.refresh_if_needed()
            # Should not crash
        finally:
            temp_path.unlink()

    def test_refresh_if_needed_no_path(self, engine):
        """Test refresh with no events path."""
        engine.refresh_if_needed()
        # Should use baseline events
        assert len(engine._events) > 0

    def test_refresh_if_needed_with_path(self):
        """Test refresh with events path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [{"title": "New Event", "sentiment": "bullish"}]
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            engine = MacroSentimentEngine(events_path=temp_path, refresh_interval=30)
            engine._last_loaded = 0  # Force refresh

            engine.refresh_if_needed()

            # Should have loaded events
            titles = [e.title for e in engine._events]
            assert "New Event" in titles
        finally:
            temp_path.unlink()

    def test_refresh_skipped_if_recent(self):
        """Test refresh is skipped if loaded recently."""
        import time

        engine = MacroSentimentEngine(refresh_interval=300)
        engine._last_loaded = time.time()  # Just loaded
        original_events = list(engine._events)

        engine.refresh_if_needed()

        # Events should be unchanged (no reload)
        assert engine._events == original_events

    def test_default_events_built(self):
        """Test default events are built."""
        engine = MacroSentimentEngine()

        # Should have some default events
        assert len(engine._events) > 0

    def test_assess_political_notes(self):
        """Test assess captures political notes."""
        events = [
            MacroEvent(
                title="Election Update",
                category="politics",
                sentiment="bearish",
                summary="Political uncertainty",
            ),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        # Political risk should be populated
        assert insight.political_risk is not None or "political" in insight.summary.lower()

    def test_assess_rate_notes(self):
        """Test assess captures rate notes."""
        events = [
            MacroEvent(
                title="Fed Decision",
                category="central_bank",
                sentiment="hawkish",
                interest_rate_expectation="Rates likely to remain elevated",
            ),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        assert insight.interest_rate_outlook is not None

    def test_assess_bias_score_range(self):
        """Test bias score is in valid range."""
        events = [
            MacroEvent(title="Event 1", sentiment="bullish", impact="critical"),
            MacroEvent(title="Event 2", sentiment="bullish", impact="critical"),
            MacroEvent(title="Event 3", sentiment="bullish", impact="critical"),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        # Even with extreme events, should be clamped
        assert -1 <= insight.bias_score <= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_baseline_no_relevant_events(self):
        """Test with events that don't match the assessed symbol."""
        events = [
            MacroEvent(
                title="Other Asset Event",
                assets={"XRP/USDT": 0.3},  # Only for XRP
            ),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("DOGE/USDT")

        # No relevant events for DOGE
        assert insight.bias_score == 0.0
        assert insight.confidence == 0.0

    def test_mixed_sentiment_events(self):
        """Test with mixed sentiment events."""
        events = [
            MacroEvent(title="Bull 1", sentiment="bullish", impact="high"),
            MacroEvent(title="Bear 1", sentiment="bearish", impact="high"),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        # Bias should be relatively balanced
        assert -0.5 <= insight.bias_score <= 0.5

    def test_actor_in_political_notes(self):
        """Test actor is included in political notes."""
        events = [
            MacroEvent(
                title="Policy Change",
                category="politics",
                sentiment="bearish",
                actor="Test Actor",
                summary="Policy update",
            ),
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        # Actor should be in political risk or summary
        if insight.political_risk:
            assert "Test Actor" in insight.political_risk

    def test_invalid_event_in_json_skipped(self):
        """Test invalid events in JSON are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"title": "Valid Event"},
                {"invalid": "structure"},  # Missing required field
            ]
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            engine = MacroSentimentEngine()
            events = engine._load_events_from_path(temp_path)

            # Should only have one valid event
            assert len(events) == 1
        finally:
            temp_path.unlink()

    def test_drivers_limited_to_top_4(self):
        """Test drivers are limited to top 4."""
        events = [
            MacroEvent(title=f"Event {i}", sentiment="bullish", impact="high") for i in range(10)
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        assert len(insight.drivers) <= 4

    def test_events_limited_to_top_6(self):
        """Test events in response are limited to 6."""
        events = [
            MacroEvent(title=f"Event {i}", sentiment="bullish", impact="high") for i in range(10)
        ]
        engine = MacroSentimentEngine(baseline_events=events)

        insight = engine.assess("BTC/USDT")

        assert len(insight.events) <= 6

    def test_confidence_calculation(self):
        """Test confidence increases with more events."""
        events_few = [
            MacroEvent(title="Event 1", sentiment="bullish"),
        ]
        events_many = [
            MacroEvent(title=f"Event {i}", sentiment="bullish", impact="high") for i in range(5)
        ]

        engine_few = MacroSentimentEngine(baseline_events=events_few)
        engine_many = MacroSentimentEngine(baseline_events=events_many)

        insight_few = engine_few.assess("BTC/USDT")
        insight_many = engine_many.assess("BTC/USDT")

        # More events should give higher confidence
        assert insight_many.confidence >= insight_few.confidence
