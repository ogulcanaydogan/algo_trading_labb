"""
Sentiment Analyzer - News and social media sentiment analysis.

Aggregates sentiment from multiple sources to generate trading signals.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sources of sentiment data."""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FEAR_GREED = "fear_greed"
    ONCHAIN = "onchain"


class SentimentLevel(Enum):
    """Sentiment classification levels."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentItem:
    """Single sentiment data point."""
    source: SentimentSource
    symbol: str
    score: float  # -1 to 1
    magnitude: float  # 0 to 1 (strength of sentiment)
    text: Optional[str]
    url: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "source": self.source.value,
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "magnitude": round(self.magnitude, 4),
            "text": self.text[:200] if self.text else None,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregateSentiment:
    """Aggregated sentiment across sources."""
    symbol: str
    overall_score: float  # -1 to 1
    overall_level: SentimentLevel
    confidence: float  # 0 to 1
    source_scores: Dict[str, float]
    source_counts: Dict[str, int]
    bullish_ratio: float
    volume_change: float  # Sentiment volume vs average
    trend: Literal["improving", "worsening", "stable"]
    signal: Literal["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]
    items: List[SentimentItem]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "overall_score": round(self.overall_score, 4),
            "overall_level": self.overall_level.value,
            "confidence": round(self.confidence, 4),
            "source_scores": {k: round(v, 4) for k, v in self.source_scores.items()},
            "source_counts": self.source_counts,
            "bullish_ratio": round(self.bullish_ratio, 4),
            "volume_change": round(self.volume_change, 4),
            "trend": self.trend,
            "signal": self.signal,
            "item_count": len(self.items),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    # Source weights
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "news": 0.35,
        "twitter": 0.25,
        "reddit": 0.20,
        "fear_greed": 0.15,
        "onchain": 0.05,
    })

    # Thresholds
    extreme_threshold: float = 0.6
    strong_threshold: float = 0.3

    # Time windows
    short_window_hours: int = 4
    long_window_hours: int = 24

    # Minimum data requirements
    min_items_for_signal: int = 5
    min_sources_for_confidence: int = 2

    # Keywords for basic sentiment
    bullish_keywords: List[str] = field(default_factory=lambda: [
        "bullish", "moon", "pump", "buy", "long", "breakout", "surge",
        "rally", "soar", "gain", "profit", "growth", "uptrend", "accumulate",
        "hodl", "btfd", "green", "rocket", "ath", "new high"
    ])

    bearish_keywords: List[str] = field(default_factory=lambda: [
        "bearish", "dump", "crash", "sell", "short", "breakdown", "plunge",
        "drop", "fall", "loss", "decline", "downtrend", "distribute",
        "rekt", "red", "correction", "fear", "panic", "capitulation"
    ])


class SentimentAnalyzer:
    """
    Analyze sentiment from multiple sources for trading signals.

    Features:
    - Multi-source aggregation (news, social, on-chain)
    - Keyword-based sentiment scoring
    - Source weighting and confidence scoring
    - Trend detection (improving/worsening)
    - Signal generation with strength levels
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._sentiment_cache: Dict[str, List[SentimentItem]] = {}
        self._historical_scores: Dict[str, List[Tuple[datetime, float]]] = {}

    def analyze_text(self, text: str, symbol: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text using keyword matching.

        Args:
            text: Text to analyze
            symbol: Symbol context

        Returns:
            Tuple of (score, magnitude)
        """
        if not text:
            return 0.0, 0.0

        text_lower = text.lower()

        # Count keyword matches
        bullish_count = sum(1 for kw in self.config.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.config.bearish_keywords if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.0

        # Calculate score (-1 to 1)
        score = (bullish_count - bearish_count) / total

        # Magnitude based on keyword density
        word_count = len(text.split())
        magnitude = min(1.0, total / max(word_count * 0.1, 1))

        return score, magnitude

    def add_sentiment_item(
        self,
        source: SentimentSource,
        symbol: str,
        text: Optional[str] = None,
        score: Optional[float] = None,
        magnitude: Optional[float] = None,
        url: Optional[str] = None,
    ) -> SentimentItem:
        """
        Add a sentiment data point.

        Args:
            source: Data source
            symbol: Asset symbol
            text: Raw text (will be analyzed if score not provided)
            score: Pre-calculated score (-1 to 1)
            magnitude: Pre-calculated magnitude (0 to 1)
            url: Source URL

        Returns:
            Created SentimentItem
        """
        # Calculate sentiment from text if not provided
        if score is None and text:
            score, magnitude = self.analyze_text(text, symbol)
        elif score is None:
            score = 0.0
            magnitude = 0.0

        if magnitude is None:
            magnitude = abs(score)

        item = SentimentItem(
            source=source,
            symbol=symbol,
            score=score,
            magnitude=magnitude,
            text=text,
            url=url,
        )

        # Cache the item
        if symbol not in self._sentiment_cache:
            self._sentiment_cache[symbol] = []
        self._sentiment_cache[symbol].append(item)

        # Cleanup old items (keep last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self._sentiment_cache[symbol] = [
            i for i in self._sentiment_cache[symbol]
            if i.timestamp > cutoff
        ]

        return item

    def add_fear_greed_index(self, symbol: str, value: float) -> SentimentItem:
        """
        Add Fear & Greed Index value.

        Args:
            symbol: Asset symbol
            value: Fear & Greed value (0-100)

        Returns:
            SentimentItem
        """
        # Convert 0-100 to -1 to 1
        score = (value - 50) / 50
        magnitude = abs(score)

        return self.add_sentiment_item(
            source=SentimentSource.FEAR_GREED,
            symbol=symbol,
            text=f"Fear & Greed Index: {value}",
            score=score,
            magnitude=magnitude,
        )

    def add_social_volume(
        self,
        source: SentimentSource,
        symbol: str,
        mentions: int,
        positive: int,
        negative: int,
        neutral: int,
    ) -> SentimentItem:
        """
        Add social media volume metrics.

        Args:
            source: Social source (twitter, reddit)
            symbol: Asset symbol
            mentions: Total mentions
            positive: Positive mentions
            negative: Negative mentions
            neutral: Neutral mentions

        Returns:
            SentimentItem
        """
        total_sentiment = positive + negative
        if total_sentiment == 0:
            score = 0.0
        else:
            score = (positive - negative) / total_sentiment

        # Magnitude based on volume
        magnitude = min(1.0, mentions / 1000)  # Normalize to typical volume

        return self.add_sentiment_item(
            source=source,
            symbol=symbol,
            text=f"Mentions: {mentions}, +{positive}/-{negative}",
            score=score,
            magnitude=magnitude,
        )

    def aggregate_sentiment(
        self,
        symbol: str,
        window_hours: Optional[int] = None,
    ) -> AggregateSentiment:
        """
        Aggregate sentiment across all sources.

        Args:
            symbol: Asset symbol
            window_hours: Time window (default: short_window)

        Returns:
            AggregateSentiment with combined analysis
        """
        window = window_hours or self.config.short_window_hours
        cutoff = datetime.now() - timedelta(hours=window)

        # Get recent items
        all_items = self._sentiment_cache.get(symbol, [])
        recent_items = [i for i in all_items if i.timestamp > cutoff]

        if not recent_items:
            return self._empty_sentiment(symbol)

        # Aggregate by source
        source_scores: Dict[str, List[float]] = {}
        source_counts: Dict[str, int] = {}

        for item in recent_items:
            source_name = item.source.value
            if source_name not in source_scores:
                source_scores[source_name] = []
                source_counts[source_name] = 0
            source_scores[source_name].append(item.score * item.magnitude)
            source_counts[source_name] += 1

        # Calculate weighted average
        weighted_sum = 0.0
        weight_total = 0.0
        avg_source_scores = {}

        for source_name, scores in source_scores.items():
            avg_score = np.mean(scores) if scores else 0
            avg_source_scores[source_name] = avg_score
            weight = self.config.source_weights.get(source_name, 0.1)
            weighted_sum += avg_score * weight
            weight_total += weight

        overall_score = weighted_sum / weight_total if weight_total > 0 else 0

        # Determine sentiment level
        if overall_score > self.config.extreme_threshold:
            level = SentimentLevel.EXTREME_GREED
        elif overall_score > self.config.strong_threshold:
            level = SentimentLevel.GREED
        elif overall_score < -self.config.extreme_threshold:
            level = SentimentLevel.EXTREME_FEAR
        elif overall_score < -self.config.strong_threshold:
            level = SentimentLevel.FEAR
        else:
            level = SentimentLevel.NEUTRAL

        # Calculate confidence
        num_sources = len(source_scores)
        num_items = len(recent_items)
        confidence = min(1.0, (
            (num_sources / self.config.min_sources_for_confidence) * 0.5 +
            (num_items / self.config.min_items_for_signal) * 0.5
        ))

        # Bullish ratio
        bullish = sum(1 for i in recent_items if i.score > 0)
        bullish_ratio = bullish / len(recent_items) if recent_items else 0.5

        # Volume change (compare to historical)
        volume_change = self._calculate_volume_change(symbol, len(recent_items))

        # Trend detection
        trend = self._detect_trend(symbol, overall_score)

        # Generate signal
        signal = self._generate_signal(overall_score, confidence, trend)

        # Store historical score
        self._store_historical_score(symbol, overall_score)

        return AggregateSentiment(
            symbol=symbol,
            overall_score=overall_score,
            overall_level=level,
            confidence=confidence,
            source_scores=avg_source_scores,
            source_counts=source_counts,
            bullish_ratio=bullish_ratio,
            volume_change=volume_change,
            trend=trend,
            signal=signal,
            items=recent_items,
        )

    def _calculate_volume_change(self, symbol: str, current_count: int) -> float:
        """Calculate volume change vs historical average."""
        historical = self._historical_scores.get(symbol, [])
        if len(historical) < 5:
            return 0.0

        # Estimate average items per period
        avg_count = len(historical) / 5  # Rough estimate
        if avg_count == 0:
            return 0.0

        return (current_count - avg_count) / avg_count

    def _detect_trend(self, symbol: str, current_score: float) -> str:
        """Detect if sentiment is improving, worsening, or stable."""
        historical = self._historical_scores.get(symbol, [])

        if len(historical) < 3:
            return "stable"

        # Get recent scores
        recent_scores = [s for _, s in historical[-5:]]
        avg_recent = np.mean(recent_scores)

        diff = current_score - avg_recent

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "worsening"
        return "stable"

    def _generate_signal(
        self,
        score: float,
        confidence: float,
        trend: str,
    ) -> str:
        """Generate trading signal from sentiment."""
        # Adjust score by confidence
        adjusted = score * confidence

        # Trend bonus
        if trend == "improving":
            adjusted += 0.1
        elif trend == "worsening":
            adjusted -= 0.1

        if adjusted > 0.5:
            return "STRONG_BUY"
        elif adjusted > 0.2:
            return "BUY"
        elif adjusted < -0.5:
            return "STRONG_SELL"
        elif adjusted < -0.2:
            return "SELL"
        return "NEUTRAL"

    def _store_historical_score(self, symbol: str, score: float):
        """Store score for trend analysis."""
        if symbol not in self._historical_scores:
            self._historical_scores[symbol] = []

        self._historical_scores[symbol].append((datetime.now(), score))

        # Keep last 100 scores
        if len(self._historical_scores[symbol]) > 100:
            self._historical_scores[symbol] = self._historical_scores[symbol][-100:]

    def _empty_sentiment(self, symbol: str) -> AggregateSentiment:
        """Return empty sentiment when no data."""
        return AggregateSentiment(
            symbol=symbol,
            overall_score=0.0,
            overall_level=SentimentLevel.NEUTRAL,
            confidence=0.0,
            source_scores={},
            source_counts={},
            bullish_ratio=0.5,
            volume_change=0.0,
            trend="stable",
            signal="NEUTRAL",
            items=[],
        )

    def get_contrarian_signal(self, symbol: str) -> Optional[str]:
        """
        Get contrarian signal (fade extreme sentiment).

        Returns:
            Signal opposite to extreme sentiment, or None
        """
        sentiment = self.aggregate_sentiment(symbol)

        # Only generate contrarian signals at extremes
        if sentiment.overall_level == SentimentLevel.EXTREME_GREED:
            return "SELL"  # Fade extreme greed
        elif sentiment.overall_level == SentimentLevel.EXTREME_FEAR:
            return "BUY"  # Fade extreme fear
        return None

    def get_momentum_signal(self, symbol: str) -> Optional[str]:
        """
        Get momentum signal (follow sentiment trend).

        Returns:
            Signal following sentiment momentum
        """
        sentiment = self.aggregate_sentiment(symbol)

        if sentiment.trend == "improving" and sentiment.overall_score > 0:
            return "BUY"
        elif sentiment.trend == "worsening" and sentiment.overall_score < 0:
            return "SELL"
        return None

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear sentiment cache."""
        if symbol:
            self._sentiment_cache.pop(symbol, None)
            self._historical_scores.pop(symbol, None)
        else:
            self._sentiment_cache.clear()
            self._historical_scores.clear()


class NewsAggregator:
    """
    Aggregate news from multiple sources.

    Integrates with:
    - NewsAPI
    - CryptoPanic
    - RSS feeds
    """

    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer
        self._api_keys: Dict[str, str] = {}

    def set_api_key(self, source: str, key: str):
        """Set API key for a news source."""
        self._api_keys[source] = key

    def process_news_item(
        self,
        symbol: str,
        title: str,
        description: str,
        url: str,
        source_name: str,
    ) -> SentimentItem:
        """
        Process a single news item.

        Args:
            symbol: Related symbol
            title: News title
            description: News description
            url: Article URL
            source_name: Source name

        Returns:
            SentimentItem with analyzed sentiment
        """
        # Combine title and description for analysis
        text = f"{title}. {description}" if description else title

        return self.analyzer.add_sentiment_item(
            source=SentimentSource.NEWS,
            symbol=symbol,
            text=text,
            url=url,
        )

    def process_crypto_panic_item(
        self,
        symbol: str,
        title: str,
        votes: Dict[str, int],
        url: str,
    ) -> SentimentItem:
        """
        Process CryptoPanic news item with vote data.

        Args:
            symbol: Crypto symbol
            title: News title
            votes: Vote counts {"positive": x, "negative": y, "important": z}
            url: Article URL

        Returns:
            SentimentItem
        """
        positive = votes.get("positive", 0)
        negative = votes.get("negative", 0)
        total = positive + negative

        if total > 0:
            score = (positive - negative) / total
            magnitude = min(1.0, total / 100)
        else:
            score, magnitude = self.analyzer.analyze_text(title, symbol)

        return self.analyzer.add_sentiment_item(
            source=SentimentSource.NEWS,
            symbol=symbol,
            text=title,
            score=score,
            magnitude=magnitude,
            url=url,
        )


class SocialMediaTracker:
    """
    Track social media sentiment.

    Integrates with:
    - Twitter/X API
    - Reddit API
    - Telegram groups
    """

    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer

    def process_tweet(
        self,
        symbol: str,
        text: str,
        likes: int,
        retweets: int,
        user_followers: int,
    ) -> SentimentItem:
        """
        Process a tweet with engagement metrics.

        Args:
            symbol: Related symbol
            text: Tweet text
            likes: Like count
            retweets: Retweet count
            user_followers: Author's followers

        Returns:
            SentimentItem
        """
        score, base_magnitude = self.analyzer.analyze_text(text, symbol)

        # Boost magnitude by engagement
        engagement = likes + retweets * 2
        influence = min(1.0, user_followers / 100000)
        magnitude = min(1.0, base_magnitude * (1 + engagement / 1000) * (1 + influence))

        return self.analyzer.add_sentiment_item(
            source=SentimentSource.TWITTER,
            symbol=symbol,
            text=text,
            score=score,
            magnitude=magnitude,
        )

    def process_reddit_post(
        self,
        symbol: str,
        title: str,
        text: str,
        upvotes: int,
        comments: int,
        subreddit: str,
    ) -> SentimentItem:
        """
        Process Reddit post.

        Args:
            symbol: Related symbol
            title: Post title
            text: Post body
            upvotes: Upvote count
            comments: Comment count
            subreddit: Subreddit name

        Returns:
            SentimentItem
        """
        full_text = f"{title}. {text}" if text else title
        score, base_magnitude = self.analyzer.analyze_text(full_text, symbol)

        # Boost by engagement
        engagement = upvotes + comments * 5
        magnitude = min(1.0, base_magnitude * (1 + engagement / 500))

        return self.analyzer.add_sentiment_item(
            source=SentimentSource.REDDIT,
            symbol=symbol,
            text=full_text[:500],
            score=score,
            magnitude=magnitude,
        )


def create_sentiment_analyzer(config: Optional[SentimentConfig] = None) -> SentimentAnalyzer:
    """Factory function to create sentiment analyzer."""
    return SentimentAnalyzer(config=config)
