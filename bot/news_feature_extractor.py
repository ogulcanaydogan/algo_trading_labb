"""
News and Economic Calendar Feature Extractor

Phase 6 of the engineering roadmap: Structured feature extraction from
news events and economic calendar data for trading signals.

This module:
1. Parses economic calendar events (FOMC, CPI, NFP, earnings, etc.)
2. Extracts sentiment from news headlines using LLM or rule-based methods
3. Computes surprise factors (actual vs expected)
4. Generates time-decay weighted features
5. Provides regime-relevant event filtering
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import re
import math

logger = logging.getLogger(__name__)


class NewsFetchThrottler:
    """Prevents excessive API calls by enforcing minimum intervals."""

    def __init__(self, min_interval_seconds: int = 300):
        """Initialize throttler.

        Args:
            min_interval_seconds: Minimum seconds between fetches for same symbol (default 5 min)
        """
        self.min_interval = min_interval_seconds
        self.last_fetch_time: Dict[str, float] = {}  # Symbol -> timestamp

    def should_fetch(self, symbol: str) -> bool:
        """Check if enough time has passed since last fetch.

        Args:
            symbol: Symbol to check

        Returns:
            True if fetch is allowed, False if still throttled
        """
        now = time.time()
        last = self.last_fetch_time.get(symbol, 0)

        if (now - last) >= self.min_interval:
            self.last_fetch_time[symbol] = now
            return True

        return False

    def get_next_fetch_time(self, symbol: str) -> float:
        """Get seconds until next fetch allowed.

        Args:
            symbol: Symbol to check

        Returns:
            Seconds until next fetch allowed (0 if immediate)
        """
        last = self.last_fetch_time.get(symbol, 0)
        next_allowed = last + self.min_interval
        return max(0, next_allowed - time.time())


class EventType(Enum):
    """Types of market-moving events."""

    # Central Bank
    FOMC = "fomc"
    ECB = "ecb"
    BOJ = "boj"
    BOE = "boe"
    FED_SPEECH = "fed_speech"

    # Economic Data
    CPI = "cpi"
    PPI = "ppi"
    NFP = "nfp"  # Non-Farm Payrolls
    UNEMPLOYMENT = "unemployment"
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    PMI = "pmi"
    ISM = "ism"
    HOUSING = "housing"
    CONSUMER_CONFIDENCE = "consumer_confidence"

    # Corporate
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"

    # Geopolitical
    ELECTION = "election"
    GEOPOLITICAL = "geopolitical"
    TRADE_WAR = "trade_war"

    # Market
    OPTIONS_EXPIRY = "options_expiry"
    FUTURES_ROLLOVER = "futures_rollover"
    INDEX_REBALANCE = "index_rebalance"

    # News
    BREAKING_NEWS = "breaking_news"
    SECTOR_NEWS = "sector_news"
    REGULATORY = "regulatory"

    # Unknown
    OTHER = "other"


class EventImpact(Enum):
    """Expected impact level of an event."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SentimentScore(Enum):
    """Sentiment classification."""

    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class EconomicEvent:
    """Represents an economic calendar event."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    title: str
    description: str = ""
    impact: EventImpact = EventImpact.MEDIUM

    # Data points
    actual: Optional[float] = None
    expected: Optional[float] = None
    previous: Optional[float] = None

    # Metadata
    currency: str = "USD"
    country: str = "US"
    source: str = ""

    # Computed fields
    surprise_factor: Optional[float] = None
    revision_factor: Optional[float] = None

    def __post_init__(self):
        """Compute derived fields after initialization."""
        self._compute_surprise()
        self._compute_revision()

    def _compute_surprise(self):
        """Compute surprise factor: (actual - expected) / abs(expected)."""
        if self.actual is not None and self.expected is not None:
            if abs(self.expected) > 0.0001:
                self.surprise_factor = (self.actual - self.expected) / abs(self.expected)
            elif self.actual != self.expected:
                self.surprise_factor = 1.0 if self.actual > self.expected else -1.0
            else:
                self.surprise_factor = 0.0

    def _compute_revision(self):
        """Compute revision factor: (expected - previous) / abs(previous)."""
        if self.expected is not None and self.previous is not None:
            if abs(self.previous) > 0.0001:
                self.revision_factor = (self.expected - self.previous) / abs(self.previous)
            elif self.expected != self.previous:
                self.revision_factor = 1.0 if self.expected > self.previous else -1.0
            else:
                self.revision_factor = 0.0


@dataclass
class NewsItem:
    """Represents a news headline/article."""

    news_id: str
    timestamp: datetime
    headline: str
    source: str

    # Optional fields
    body: str = ""
    url: str = ""
    tickers: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)

    # Sentiment analysis
    sentiment_score: Optional[float] = None  # -1 to 1
    sentiment_confidence: float = 0.0
    sentiment_source: str = ""  # "llm", "rule_based", "finbert", etc.

    # Classification
    event_type: EventType = EventType.OTHER
    impact: EventImpact = EventImpact.MEDIUM

    # Keywords extracted
    keywords: List[str] = field(default_factory=list)


@dataclass
class NewsFeatures:
    """Extracted features from news and events."""

    timestamp: datetime

    # Aggregate sentiment
    overall_sentiment: float = 0.0
    sentiment_momentum: float = 0.0  # Change in sentiment
    sentiment_dispersion: float = 0.0  # Variance in sentiment

    # Event counts (in lookback window)
    high_impact_events: int = 0
    medium_impact_events: int = 0
    low_impact_events: int = 0

    # Surprise aggregates
    avg_surprise: float = 0.0
    max_positive_surprise: float = 0.0
    max_negative_surprise: float = 0.0

    # Category-specific features
    central_bank_sentiment: float = 0.0
    economic_data_sentiment: float = 0.0
    corporate_sentiment: float = 0.0
    geopolitical_sentiment: float = 0.0

    # Time-weighted features
    decay_weighted_sentiment: float = 0.0
    recent_news_intensity: float = 0.0

    # Upcoming events
    hours_to_next_high_impact: Optional[float] = None
    next_event_type: Optional[str] = None

    # Risk indicators
    news_velocity: float = 0.0  # News items per hour
    headline_risk_score: float = 0.0  # Aggregate risk from headlines

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return [
            self.overall_sentiment,
            self.sentiment_momentum,
            self.sentiment_dispersion,
            float(self.high_impact_events),
            float(self.medium_impact_events),
            float(self.low_impact_events),
            self.avg_surprise,
            self.max_positive_surprise,
            self.max_negative_surprise,
            self.central_bank_sentiment,
            self.economic_data_sentiment,
            self.corporate_sentiment,
            self.geopolitical_sentiment,
            self.decay_weighted_sentiment,
            self.recent_news_intensity,
            self.hours_to_next_high_impact or 168.0,  # Default to 1 week
            self.news_velocity,
            self.headline_risk_score,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Return names of features in the vector."""
        return [
            "overall_sentiment",
            "sentiment_momentum",
            "sentiment_dispersion",
            "high_impact_events",
            "medium_impact_events",
            "low_impact_events",
            "avg_surprise",
            "max_positive_surprise",
            "max_negative_surprise",
            "central_bank_sentiment",
            "economic_data_sentiment",
            "corporate_sentiment",
            "geopolitical_sentiment",
            "decay_weighted_sentiment",
            "recent_news_intensity",
            "hours_to_next_high_impact",
            "news_velocity",
            "headline_risk_score",
        ]


class SentimentAnalyzer(ABC):
    """Base class for sentiment analysis."""

    @abstractmethod
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (bearish) to 1 (bullish)
            confidence: 0 to 1
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analyzer."""
        pass


class RuleBasedSentimentAnalyzer(SentimentAnalyzer):
    """Rule-based sentiment analyzer using keyword matching."""

    # Keyword dictionaries with weights
    BULLISH_KEYWORDS = {
        # Strong bullish
        "surge": 0.8,
        "soar": 0.8,
        "rally": 0.7,
        "breakout": 0.7,
        "boom": 0.7,
        "skyrocket": 0.9,
        "moon": 0.6,
        # Moderate bullish
        "rise": 0.5,
        "gain": 0.5,
        "up": 0.4,
        "higher": 0.4,
        "advance": 0.5,
        "climb": 0.5,
        "jump": 0.6,
        "pop": 0.5,
        # Mild bullish
        "positive": 0.3,
        "bullish": 0.4,
        "optimistic": 0.3,
        "growth": 0.3,
        "expand": 0.3,
        "improve": 0.3,
        # Beat expectations
        "beat": 0.6,
        "exceed": 0.5,
        "surpass": 0.5,
        "outperform": 0.5,
        "strong": 0.4,
        "robust": 0.4,
    }

    BEARISH_KEYWORDS = {
        # Strong bearish
        "crash": -0.9,
        "plunge": -0.8,
        "collapse": -0.8,
        "tank": -0.7,
        "tumble": -0.7,
        "plummet": -0.8,
        "selloff": -0.7,
        # Moderate bearish
        "fall": -0.5,
        "drop": -0.5,
        "down": -0.4,
        "lower": -0.4,
        "decline": -0.5,
        "slide": -0.5,
        "sink": -0.6,
        "dump": -0.6,
        # Mild bearish
        "negative": -0.3,
        "bearish": -0.4,
        "pessimistic": -0.3,
        "weak": -0.3,
        "slowdown": -0.3,
        "contract": -0.3,
        # Miss expectations
        "miss": -0.6,
        "disappoint": -0.5,
        "underperform": -0.5,
        "fail": -0.5,
        "concern": -0.3,
        "worry": -0.3,
        # Risk words
        "risk": -0.3,
        "fear": -0.4,
        "panic": -0.6,
        "crisis": -0.7,
        "recession": -0.6,
        "inflation": -0.4,
        "default": -0.7,
    }

    INTENSITY_MODIFIERS = {
        "very": 1.3,
        "extremely": 1.5,
        "significantly": 1.2,
        "slightly": 0.7,
        "somewhat": 0.8,
        "marginally": 0.6,
        "sharply": 1.4,
        "dramatically": 1.4,
        "massively": 1.5,
    }

    @property
    def name(self) -> str:
        return "rule_based"

    def analyze(self, text: str) -> Tuple[float, float]:
        """Analyze text using keyword matching."""
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        total_score = 0.0
        match_count = 0

        # Check for intensity modifiers
        intensity_multiplier = 1.0
        for modifier, mult in self.INTENSITY_MODIFIERS.items():
            if modifier in text_lower:
                intensity_multiplier = max(intensity_multiplier, mult)

        # Score bullish keywords
        for word, weight in self.BULLISH_KEYWORDS.items():
            if word in words:
                total_score += weight * intensity_multiplier
                match_count += 1

        # Score bearish keywords
        for word, weight in self.BEARISH_KEYWORDS.items():
            if word in words:
                total_score += weight * intensity_multiplier
                match_count += 1

        if match_count == 0:
            return 0.0, 0.0

        # Normalize score to -1 to 1 range
        avg_score = total_score / match_count
        final_score = max(-1.0, min(1.0, avg_score))

        # Confidence based on number of matches and intensity
        confidence = min(1.0, match_count / 5.0) * 0.7 + 0.3 * abs(final_score)

        return final_score, confidence


class LLMSentimentAnalyzer(SentimentAnalyzer):
    """LLM-based sentiment analyzer."""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize with optional LLM client.

        Args:
            llm_client: LLM client with analyze_sentiment method
        """
        self.llm_client = llm_client
        self._fallback = RuleBasedSentimentAnalyzer()

    @property
    def name(self) -> str:
        return "llm"

    def analyze(self, text: str) -> Tuple[float, float]:
        """Analyze using LLM, with rule-based fallback."""
        if self.llm_client is None:
            return self._fallback.analyze(text)

        try:
            # Expected format from LLM: {"sentiment": float, "confidence": float}
            result = self.llm_client.analyze_sentiment(text)

            if isinstance(result, dict):
                sentiment = result.get("sentiment", 0.0)
                confidence = result.get("confidence", 0.5)
                return float(sentiment), float(confidence)

            return self._fallback.analyze(text)

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}, using fallback")
            return self._fallback.analyze(text)


class NewsFeatureExtractor:
    """
    Main class for extracting trading features from news and economic events.

    Features:
    - Aggregates sentiment from multiple news sources
    - Computes surprise factors from economic releases
    - Applies time decay to older events
    - Provides regime-relevant filtering
    - Generates ML-ready feature vectors
    """

    # Event type to category mapping
    EVENT_CATEGORIES = {
        EventType.FOMC: "central_bank",
        EventType.ECB: "central_bank",
        EventType.BOJ: "central_bank",
        EventType.BOE: "central_bank",
        EventType.FED_SPEECH: "central_bank",
        EventType.CPI: "economic_data",
        EventType.PPI: "economic_data",
        EventType.NFP: "economic_data",
        EventType.UNEMPLOYMENT: "economic_data",
        EventType.GDP: "economic_data",
        EventType.RETAIL_SALES: "economic_data",
        EventType.PMI: "economic_data",
        EventType.ISM: "economic_data",
        EventType.HOUSING: "economic_data",
        EventType.CONSUMER_CONFIDENCE: "economic_data",
        EventType.EARNINGS: "corporate",
        EventType.GUIDANCE: "corporate",
        EventType.DIVIDEND: "corporate",
        EventType.STOCK_SPLIT: "corporate",
        EventType.ELECTION: "geopolitical",
        EventType.GEOPOLITICAL: "geopolitical",
        EventType.TRADE_WAR: "geopolitical",
    }

    # Risk keywords for headline risk scoring
    RISK_KEYWORDS = [
        "crash",
        "crisis",
        "collapse",
        "panic",
        "fear",
        "recession",
        "default",
        "bankruptcy",
        "war",
        "conflict",
        "sanctions",
        "hack",
        "breach",
        "fraud",
        "investigation",
        "lawsuit",
    ]

    def __init__(
        self,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        lookback_hours: int = 24,
        decay_half_life_hours: float = 6.0,
        fetch_throttle_minutes: int = 5,
    ):
        """
        Initialize the feature extractor.

        Args:
            sentiment_analyzer: Sentiment analyzer to use
            lookback_hours: Hours to look back for news aggregation
            decay_half_life_hours: Half-life for time decay weighting
            fetch_throttle_minutes: Minimum minutes between API fetches per symbol (default 5)
        """
        self.sentiment_analyzer = sentiment_analyzer or RuleBasedSentimentAnalyzer()
        self.lookback_hours = lookback_hours
        self.decay_half_life_hours = decay_half_life_hours

        # Storage for events and news
        self.economic_events: List[EconomicEvent] = []
        self.news_items: List[NewsItem] = []
        self.upcoming_events: List[EconomicEvent] = []

        # Cache for computed features
        self._feature_cache: Dict[str, NewsFeatures] = {}
        self._cache_ttl_minutes = 5

        # Fetch throttler to reduce API calls
        self.fetch_throttler = NewsFetchThrottler(min_interval_seconds=fetch_throttle_minutes * 60)

        logger.info(
            f"NewsFeatureExtractor initialized with {self.sentiment_analyzer.name} analyzer"
        )
        logger.info(
            f"News fetch throttled to 1 request per {fetch_throttle_minutes} minutes per symbol"
        )

    def add_economic_event(self, event: EconomicEvent):
        """Add an economic event to the tracker."""
        self.economic_events.append(event)
        self._invalidate_cache()
        logger.debug(f"Added economic event: {event.event_type.value} - {event.title}")

    def add_news_item(self, news: NewsItem):
        """Add a news item and analyze sentiment if not already done."""
        if news.sentiment_score is None:
            score, confidence = self.sentiment_analyzer.analyze(news.headline)
            news.sentiment_score = score
            news.sentiment_confidence = confidence
            news.sentiment_source = self.sentiment_analyzer.name

        self.news_items.append(news)
        self._invalidate_cache()
        logger.debug(f"Added news: {news.headline[:50]}... sentiment={news.sentiment_score:.2f}")

    def add_upcoming_event(self, event: EconomicEvent):
        """Add an upcoming scheduled event."""
        self.upcoming_events.append(event)
        self.upcoming_events.sort(key=lambda e: e.timestamp)
        logger.debug(f"Added upcoming event: {event.event_type.value} at {event.timestamp}")

    def _invalidate_cache(self):
        """Invalidate the feature cache."""
        self._feature_cache.clear()

    def _compute_time_decay(self, event_time: datetime, current_time: datetime) -> float:
        """Compute exponential time decay weight."""
        hours_ago = (current_time - event_time).total_seconds() / 3600
        if hours_ago < 0:
            return 0.0

        # Exponential decay: weight = 2^(-t/half_life)
        decay = math.pow(2, -hours_ago / self.decay_half_life_hours)
        return decay

    def _filter_by_lookback(
        self, items: List[Any], current_time: datetime, timestamp_attr: str = "timestamp"
    ) -> List[Any]:
        """Filter items to only those within lookback window."""
        cutoff = current_time - timedelta(hours=self.lookback_hours)
        return [item for item in items if getattr(item, timestamp_attr) >= cutoff]

    def extract_features(
        self, current_time: Optional[datetime] = None, regime: Optional[str] = None
    ) -> NewsFeatures:
        """
        Extract aggregated features from news and events.

        Args:
            current_time: Reference time for feature extraction
            regime: Current market regime for regime-specific filtering

        Returns:
            NewsFeatures with all extracted features
        """
        current_time = current_time or datetime.now()
        cache_key = f"{current_time.isoformat()}_{regime}"

        # Check cache
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # Filter to lookback window
        recent_events = self._filter_by_lookback(self.economic_events, current_time)
        recent_news = self._filter_by_lookback(self.news_items, current_time)

        features = NewsFeatures(timestamp=current_time)

        # Compute sentiment features
        self._compute_sentiment_features(features, recent_news, current_time)

        # Compute event features
        self._compute_event_features(features, recent_events, current_time)

        # Compute category-specific sentiment
        self._compute_category_sentiment(features, recent_news, recent_events, current_time)

        # Compute upcoming event features
        self._compute_upcoming_features(features, current_time)

        # Compute risk indicators
        self._compute_risk_indicators(features, recent_news, current_time)

        # Cache result
        self._feature_cache[cache_key] = features

        return features

    def _compute_sentiment_features(
        self, features: NewsFeatures, news_items: List[NewsItem], current_time: datetime
    ):
        """Compute aggregate sentiment features."""
        if not news_items:
            return

        sentiments = []
        weighted_sentiments = []
        weights_sum = 0.0

        for news in news_items:
            if news.sentiment_score is not None:
                sentiments.append(news.sentiment_score)

                # Time-decay weighted
                decay = self._compute_time_decay(news.timestamp, current_time)
                weight = decay * news.sentiment_confidence
                weighted_sentiments.append(news.sentiment_score * weight)
                weights_sum += weight

        if sentiments:
            features.overall_sentiment = sum(sentiments) / len(sentiments)

            if len(sentiments) > 1:
                mean = features.overall_sentiment
                variance = sum((s - mean) ** 2 for s in sentiments) / len(sentiments)
                features.sentiment_dispersion = math.sqrt(variance)

        if weights_sum > 0:
            features.decay_weighted_sentiment = sum(weighted_sentiments) / weights_sum

        # Compute sentiment momentum (recent vs older)
        if len(news_items) >= 4:
            sorted_news = sorted(news_items, key=lambda n: n.timestamp, reverse=True)
            half = len(sorted_news) // 2

            recent_avg = (
                sum(n.sentiment_score for n in sorted_news[:half] if n.sentiment_score is not None)
                / half
            )
            older_avg = (
                sum(n.sentiment_score for n in sorted_news[half:] if n.sentiment_score is not None)
                / half
            )

            features.sentiment_momentum = recent_avg - older_avg

    def _compute_event_features(
        self, features: NewsFeatures, events: List[EconomicEvent], current_time: datetime
    ):
        """Compute features from economic events."""
        if not events:
            return

        surprises = []

        for event in events:
            # Count by impact
            if event.impact == EventImpact.HIGH or event.impact == EventImpact.CRITICAL:
                features.high_impact_events += 1
            elif event.impact == EventImpact.MEDIUM:
                features.medium_impact_events += 1
            else:
                features.low_impact_events += 1

            # Collect surprises
            if event.surprise_factor is not None:
                surprises.append(event.surprise_factor)

        if surprises:
            features.avg_surprise = sum(surprises) / len(surprises)
            features.max_positive_surprise = max(surprises)
            features.max_negative_surprise = min(surprises)

    def _compute_category_sentiment(
        self,
        features: NewsFeatures,
        news_items: List[NewsItem],
        events: List[EconomicEvent],
        current_time: datetime,
    ):
        """Compute category-specific sentiment."""
        category_sentiments: Dict[str, List[float]] = {
            "central_bank": [],
            "economic_data": [],
            "corporate": [],
            "geopolitical": [],
        }

        # From news items
        for news in news_items:
            if news.sentiment_score is not None:
                category = self.EVENT_CATEGORIES.get(news.event_type, "other")
                if category in category_sentiments:
                    category_sentiments[category].append(news.sentiment_score)

        # From events (use surprise as sentiment proxy)
        for event in events:
            if event.surprise_factor is not None:
                category = self.EVENT_CATEGORIES.get(event.event_type, "other")
                if category in category_sentiments:
                    # Normalize surprise to -1 to 1 range
                    normalized = max(-1, min(1, event.surprise_factor))
                    category_sentiments[category].append(normalized)

        # Compute averages
        if category_sentiments["central_bank"]:
            features.central_bank_sentiment = sum(category_sentiments["central_bank"]) / len(
                category_sentiments["central_bank"]
            )

        if category_sentiments["economic_data"]:
            features.economic_data_sentiment = sum(category_sentiments["economic_data"]) / len(
                category_sentiments["economic_data"]
            )

        if category_sentiments["corporate"]:
            features.corporate_sentiment = sum(category_sentiments["corporate"]) / len(
                category_sentiments["corporate"]
            )

        if category_sentiments["geopolitical"]:
            features.geopolitical_sentiment = sum(category_sentiments["geopolitical"]) / len(
                category_sentiments["geopolitical"]
            )

    def _compute_upcoming_features(self, features: NewsFeatures, current_time: datetime):
        """Compute features related to upcoming events."""
        # Filter to future events
        future_events = [e for e in self.upcoming_events if e.timestamp > current_time]

        if not future_events:
            return

        # Find next high-impact event
        high_impact_events = [
            e for e in future_events if e.impact in [EventImpact.HIGH, EventImpact.CRITICAL]
        ]

        if high_impact_events:
            next_event = high_impact_events[0]
            hours_until = (next_event.timestamp - current_time).total_seconds() / 3600
            features.hours_to_next_high_impact = hours_until
            features.next_event_type = next_event.event_type.value

    def _compute_risk_indicators(
        self, features: NewsFeatures, news_items: List[NewsItem], current_time: datetime
    ):
        """Compute risk-related indicators."""
        if not news_items:
            return

        # News velocity (items per hour)
        if news_items:
            time_span_hours = self.lookback_hours
            features.news_velocity = len(news_items) / time_span_hours

        # Recent news intensity (last hour weighted more)
        recent_cutoff = current_time - timedelta(hours=1)
        recent_count = sum(1 for n in news_items if n.timestamp >= recent_cutoff)
        features.recent_news_intensity = recent_count / max(1, len(news_items))

        # Headline risk score
        risk_score = 0.0
        for news in news_items:
            headline_lower = news.headline.lower()
            for keyword in self.RISK_KEYWORDS:
                if keyword in headline_lower:
                    decay = self._compute_time_decay(news.timestamp, current_time)
                    risk_score += decay
                    break

        features.headline_risk_score = min(1.0, risk_score / 5.0)

    def get_regime_relevant_features(
        self, regime: str, current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get features most relevant to a specific regime.

        Args:
            regime: Market regime (trending_bullish, trending_bearish, mean_reverting, etc.)
            current_time: Reference time

        Returns:
            Dictionary of regime-relevant features
        """
        features = self.extract_features(current_time, regime)

        regime_features = {
            "overall_sentiment": features.overall_sentiment,
            "sentiment_momentum": features.sentiment_momentum,
            "headline_risk": features.headline_risk_score,
        }

        # Add regime-specific features
        if "trend" in regime.lower():
            regime_features["central_bank_sentiment"] = features.central_bank_sentiment
            regime_features["economic_sentiment"] = features.economic_data_sentiment
            regime_features["news_momentum"] = features.sentiment_momentum

        elif "mean_revert" in regime.lower():
            regime_features["sentiment_dispersion"] = features.sentiment_dispersion
            regime_features["decay_weighted"] = features.decay_weighted_sentiment

        elif "volatile" in regime.lower() or "crisis" in regime.lower():
            regime_features["geopolitical_sentiment"] = features.geopolitical_sentiment
            regime_features["high_impact_count"] = float(features.high_impact_events)
            regime_features["max_surprise"] = max(
                abs(features.max_positive_surprise), abs(features.max_negative_surprise)
            )

        return regime_features

    def cleanup_old_data(self, keep_hours: int = 72):
        """Remove data older than keep_hours."""
        cutoff = datetime.now() - timedelta(hours=keep_hours)

        old_events = len(self.economic_events)
        self.economic_events = [e for e in self.economic_events if e.timestamp >= cutoff]

        old_news = len(self.news_items)
        self.news_items = [n for n in self.news_items if n.timestamp >= cutoff]

        # Clean up past upcoming events
        now = datetime.now()
        self.upcoming_events = [e for e in self.upcoming_events if e.timestamp > now]

        self._invalidate_cache()

        logger.info(
            f"Cleanup: removed {old_events - len(self.economic_events)} events, "
            f"{old_news - len(self.news_items)} news items"
        )


class EconomicCalendarParser:
    """Parser for economic calendar data from various sources."""

    # Event title patterns to event types
    EVENT_PATTERNS = {
        r"fomc|federal reserve|fed meeting": EventType.FOMC,
        r"ecb|european central bank": EventType.ECB,
        r"boj|bank of japan": EventType.BOJ,
        r"boe|bank of england": EventType.BOE,
        r"fed\s+speak|fed\s+chair": EventType.FED_SPEECH,
        r"cpi|consumer price|inflation": EventType.CPI,
        r"ppi|producer price": EventType.PPI,
        r"nonfarm|non-farm|payroll": EventType.NFP,
        r"unemployment": EventType.UNEMPLOYMENT,
        r"gdp|gross domestic": EventType.GDP,
        r"retail sales": EventType.RETAIL_SALES,
        r"pmi|purchasing manager": EventType.PMI,
        r"ism\s+manufacturing|ism\s+services": EventType.ISM,
        r"housing|home sales|building permit": EventType.HOUSING,
        r"consumer confidence|sentiment": EventType.CONSUMER_CONFIDENCE,
        r"earnings|quarterly report": EventType.EARNINGS,
        r"guidance|outlook": EventType.GUIDANCE,
        r"election|vote": EventType.ELECTION,
    }

    # Impact keywords
    HIGH_IMPACT_PATTERNS = [r"fomc", r"fed", r"cpi", r"nfp", r"gdp", r"ecb", r"boj"]

    MEDIUM_IMPACT_PATTERNS = [r"pmi", r"ism", r"retail", r"housing", r"unemployment"]

    def parse_event(
        self,
        title: str,
        timestamp: datetime,
        actual: Optional[float] = None,
        expected: Optional[float] = None,
        previous: Optional[float] = None,
        source: str = "",
    ) -> EconomicEvent:
        """
        Parse an economic event from raw data.

        Args:
            title: Event title
            timestamp: Event time
            actual: Actual value (if released)
            expected: Expected/consensus value
            previous: Previous period value
            source: Data source name

        Returns:
            Parsed EconomicEvent
        """
        event_type = self._detect_event_type(title)
        impact = self._estimate_impact(title, event_type)

        event_id = f"{event_type.value}_{timestamp.strftime('%Y%m%d_%H%M')}"

        return EconomicEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            title=title,
            impact=impact,
            actual=actual,
            expected=expected,
            previous=previous,
            source=source,
        )

    def _detect_event_type(self, title: str) -> EventType:
        """Detect event type from title."""
        title_lower = title.lower()

        for pattern, event_type in self.EVENT_PATTERNS.items():
            if re.search(pattern, title_lower):
                return event_type

        return EventType.OTHER

    def _estimate_impact(self, title: str, event_type: EventType) -> EventImpact:
        """Estimate event impact level."""
        title_lower = title.lower()

        # Check high impact patterns
        for pattern in self.HIGH_IMPACT_PATTERNS:
            if re.search(pattern, title_lower):
                return EventImpact.HIGH

        # Check medium impact patterns
        for pattern in self.MEDIUM_IMPACT_PATTERNS:
            if re.search(pattern, title_lower):
                return EventImpact.MEDIUM

        # Default by event type
        high_impact_types = {
            EventType.FOMC,
            EventType.ECB,
            EventType.CPI,
            EventType.NFP,
            EventType.GDP,
        }

        if event_type in high_impact_types:
            return EventImpact.HIGH

        return EventImpact.LOW

    def parse_from_json(self, json_data: Dict[str, Any]) -> EconomicEvent:
        """Parse event from JSON format (common API format)."""
        return self.parse_event(
            title=json_data.get("title", json_data.get("name", "")),
            timestamp=datetime.fromisoformat(
                json_data.get("datetime", json_data.get("timestamp", ""))
            ),
            actual=json_data.get("actual"),
            expected=json_data.get("expected", json_data.get("forecast")),
            previous=json_data.get("previous"),
            source=json_data.get("source", ""),
        )


class NewsParser:
    """Parser for news data from various sources."""

    def __init__(self, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """Initialize with optional sentiment analyzer."""
        self.sentiment_analyzer = sentiment_analyzer or RuleBasedSentimentAnalyzer()

    def parse_headline(
        self,
        headline: str,
        timestamp: datetime,
        source: str = "",
        tickers: Optional[List[str]] = None,
        body: str = "",
    ) -> NewsItem:
        """
        Parse a news headline.

        Args:
            headline: News headline
            timestamp: Publication time
            source: News source name
            tickers: Related tickers
            body: Optional article body

        Returns:
            Parsed NewsItem with sentiment analysis
        """
        news_id = f"news_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(headline) % 10000}"

        # Analyze sentiment
        sentiment, confidence = self.sentiment_analyzer.analyze(headline)

        # Detect event type from headline
        event_type = self._detect_news_type(headline)

        # Estimate impact
        impact = self._estimate_impact(headline, event_type)

        # Extract keywords
        keywords = self._extract_keywords(headline)

        return NewsItem(
            news_id=news_id,
            timestamp=timestamp,
            headline=headline,
            source=source,
            body=body,
            tickers=tickers or [],
            sentiment_score=sentiment,
            sentiment_confidence=confidence,
            sentiment_source=self.sentiment_analyzer.name,
            event_type=event_type,
            impact=impact,
            keywords=keywords,
        )

    def _detect_news_type(self, headline: str) -> EventType:
        """Detect news type from headline."""
        headline_lower = headline.lower()

        if any(w in headline_lower for w in ["fed", "fomc", "powell", "central bank"]):
            return EventType.FED_SPEECH

        if any(w in headline_lower for w in ["earnings", "quarter", "profit", "revenue"]):
            return EventType.EARNINGS

        if any(w in headline_lower for w in ["war", "conflict", "sanction", "tariff"]):
            return EventType.GEOPOLITICAL

        if any(w in headline_lower for w in ["regulation", "sec", "lawsuit", "fine"]):
            return EventType.REGULATORY

        if any(w in headline_lower for w in ["breaking", "urgent", "alert"]):
            return EventType.BREAKING_NEWS

        return EventType.OTHER

    def _estimate_impact(self, headline: str, event_type: EventType) -> EventImpact:
        """Estimate news impact level."""
        headline_lower = headline.lower()

        # Breaking news indicators
        if any(w in headline_lower for w in ["breaking", "urgent", "alert", "crash", "crisis"]):
            return EventImpact.HIGH

        # Fed/central bank
        if event_type == EventType.FED_SPEECH:
            return EventImpact.HIGH

        # Earnings from major companies
        if event_type == EventType.EARNINGS:
            major_companies = ["apple", "microsoft", "amazon", "google", "nvidia", "tesla"]
            if any(c in headline_lower for c in major_companies):
                return EventImpact.HIGH
            return EventImpact.MEDIUM

        return EventImpact.LOW

    def _extract_keywords(self, headline: str) -> List[str]:
        """Extract significant keywords from headline."""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r"\b[A-Z][a-z]+\b|\b[A-Z]+\b", headline)

        # Filter common words
        stop_words = {"The", "A", "An", "In", "On", "At", "To", "For", "And", "Or", "Is", "Are"}
        keywords = [w for w in words if w not in stop_words]

        return keywords[:10]  # Limit to 10 keywords

    def parse_from_json(self, json_data: Dict[str, Any]) -> NewsItem:
        """Parse news from JSON format (common API format)."""
        return self.parse_headline(
            headline=json_data.get("headline", json_data.get("title", "")),
            timestamp=datetime.fromisoformat(
                json_data.get(
                    "datetime", json_data.get("publishedAt", json_data.get("timestamp", ""))
                )
            ),
            source=json_data.get("source", ""),
            tickers=json_data.get("tickers", json_data.get("symbols", [])),
            body=json_data.get("body", json_data.get("description", "")),
        )


# Factory function for easy initialization
def create_news_feature_extractor(
    use_llm: bool = False,
    llm_client: Optional[Any] = None,
    lookback_hours: int = 24,
    decay_half_life_hours: float = 6.0,
) -> NewsFeatureExtractor:
    """
    Create a news feature extractor with appropriate configuration.

    Args:
        use_llm: Whether to use LLM for sentiment analysis
        llm_client: LLM client if using LLM
        lookback_hours: Lookback window for feature aggregation
        decay_half_life_hours: Half-life for time decay

    Returns:
        Configured NewsFeatureExtractor
    """
    if use_llm:
        analyzer = LLMSentimentAnalyzer(llm_client)
    else:
        analyzer = RuleBasedSentimentAnalyzer()

    return NewsFeatureExtractor(
        sentiment_analyzer=analyzer,
        lookback_hours=lookback_hours,
        decay_half_life_hours=decay_half_life_hours,
    )


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    # Create extractor
    extractor = create_news_feature_extractor(
        use_llm=False, lookback_hours=24, decay_half_life_hours=6.0
    )

    # Add some sample events
    from datetime import datetime, timedelta

    now = datetime.now()

    # Economic events
    parser = EconomicCalendarParser()

    events = [
        {
            "title": "FOMC Interest Rate Decision",
            "datetime": (now - timedelta(hours=2)).isoformat(),
            "actual": 5.25,
            "expected": 5.25,
            "previous": 5.0,
        },
        {
            "title": "CPI Inflation YoY",
            "datetime": (now - timedelta(hours=6)).isoformat(),
            "actual": 3.2,
            "expected": 3.1,
            "previous": 3.0,
        },
    ]

    for event_data in events:
        event = parser.parse_from_json(event_data)
        extractor.add_economic_event(event)

    # News items
    news_parser = NewsParser()

    headlines = [
        "Fed Chair Powell signals potential rate cuts ahead",
        "Tech stocks surge on strong earnings reports",
        "Bitcoin crashes 10% amid regulatory concerns",
        "Nvidia beats earnings expectations, stock jumps 5%",
    ]

    for i, headline in enumerate(headlines):
        news = news_parser.parse_headline(
            headline=headline, timestamp=now - timedelta(hours=i), source="Reuters"
        )
        extractor.add_news_item(news)

    # Add upcoming event
    upcoming = parser.parse_event(
        title="NFP Employment Report",
        timestamp=now + timedelta(hours=48),
        expected=200000,
        previous=180000,
    )
    extractor.add_upcoming_event(upcoming)

    # Extract features
    features = extractor.extract_features()

    print("\n=== News Feature Extraction Demo ===")
    print(f"Overall sentiment: {features.overall_sentiment:.3f}")
    print(f"Sentiment momentum: {features.sentiment_momentum:.3f}")
    print(f"Decay-weighted sentiment: {features.decay_weighted_sentiment:.3f}")
    print(f"High impact events: {features.high_impact_events}")
    print(f"Avg surprise: {features.avg_surprise:.3f}")
    print(f"Central bank sentiment: {features.central_bank_sentiment:.3f}")
    print(f"Hours to next high impact: {features.hours_to_next_high_impact:.1f}")
    print(f"News velocity: {features.news_velocity:.2f} items/hour")
    print(f"Headline risk score: {features.headline_risk_score:.3f}")

    print("\nFeature vector:")
    print(f"  Names: {features.feature_names()[:6]}...")
    print(f"  Values: {[f'{v:.3f}' for v in features.to_vector()[:6]]}...")

    # Regime-specific features
    print("\nRegime-relevant features (trending_bullish):")
    regime_features = extractor.get_regime_relevant_features("trending_bullish")
    for name, value in regime_features.items():
        print(f"  {name}: {value:.3f}")
