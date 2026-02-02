"""
Real-Time News Reactor.

Sub-second reaction to market news:
- WebSocket news feeds integration
- Streaming NLP classification
- Urgency scoring (1-10)
- Auto-action (close, hedge, reduce size)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
import re

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """A news event."""
    id: str
    timestamp: datetime
    headline: str
    source: str
    symbols: List[str]  # Affected symbols
    sentiment: float  # -1 to 1
    urgency: int  # 1-10
    category: str  # earnings, macro, regulatory, technical, etc.
    raw_content: str = ""

    @property
    def is_critical(self) -> bool:
        return self.urgency >= 8

    @property
    def is_actionable(self) -> bool:
        return self.urgency >= 6 and abs(self.sentiment) >= 0.3


@dataclass
class NewsAction:
    """Recommended action from news."""
    action: str  # CLOSE_ALL, CLOSE_SYMBOL, HEDGE, REDUCE_SIZE, ALERT, NONE
    symbols: List[str]
    urgency: int
    reason: str
    sentiment: float
    time_sensitive_seconds: int = 300  # Action valid for 5 minutes


class RealTimeNewsReactor:
    """
    Reacts to news in real-time.

    Features:
    - Keyword-based urgency detection
    - Sentiment analysis
    - Auto-action generation
    - Cooldown to prevent over-reaction
    """

    # Critical keywords that warrant immediate action
    CRITICAL_KEYWORDS = {
        "hack": 9,
        "hacked": 9,
        "exploit": 9,
        "security breach": 9,
        "bankruptcy": 9,
        "insolvent": 9,
        "sec charges": 8,
        "sec lawsuit": 8,
        "fraud": 8,
        "investigation": 7,
        "crash": 8,
        "flash crash": 9,
        "circuit breaker": 8,
        "trading halt": 8,
        "emergency": 8,
        "crisis": 7,
        "default": 8,
        "bank run": 9,
        "liquidity crisis": 8,
    }

    # Positive keywords
    POSITIVE_KEYWORDS = {
        "approval": 0.6,
        "approved": 0.6,
        "partnership": 0.4,
        "adoption": 0.5,
        "bullish": 0.4,
        "upgrade": 0.3,
        "record high": 0.5,
        "etf approved": 0.7,
        "institutional": 0.4,
    }

    # Negative keywords
    NEGATIVE_KEYWORDS = {
        "bearish": -0.4,
        "downgrade": -0.3,
        "lawsuit": -0.5,
        "regulatory": -0.3,
        "ban": -0.6,
        "prohibited": -0.5,
        "warning": -0.3,
        "concern": -0.2,
        "selloff": -0.5,
        "dump": -0.4,
    }

    def __init__(
        self,
        action_callback: Optional[Callable[[NewsAction], None]] = None,
        cooldown_seconds: int = 60,
        max_events_per_minute: int = 10,
    ):
        self.action_callback = action_callback
        self.cooldown_seconds = cooldown_seconds
        self.max_events_per_minute = max_events_per_minute

        # Event tracking
        self._recent_events: deque = deque(maxlen=100)
        self._last_action_time: Dict[str, datetime] = {}

        # Stats
        self._events_processed = 0
        self._actions_triggered = 0

        logger.info("RealTimeNewsReactor initialized")

    async def process_news(self, news: NewsEvent) -> Optional[NewsAction]:
        """
        Process a news event and determine action.

        Args:
            news: NewsEvent to process

        Returns:
            NewsAction if action needed, None otherwise
        """
        self._events_processed += 1
        self._recent_events.append(news)

        # Check rate limiting
        if self._is_rate_limited():
            logger.warning("News reactor rate limited")
            return None

        # Enhance news with analysis
        news = self._analyze_news(news)

        # Check if actionable
        if not news.is_actionable:
            logger.debug(f"News not actionable: {news.headline[:50]}")
            return None

        # Check cooldown
        for symbol in news.symbols:
            if self._is_in_cooldown(symbol):
                logger.debug(f"Symbol {symbol} in cooldown")
                return None

        # Generate action
        action = self._generate_action(news)

        if action and action.action != "NONE":
            self._actions_triggered += 1

            # Set cooldown
            for symbol in action.symbols:
                self._last_action_time[symbol] = datetime.now()

            # Trigger callback
            if self.action_callback:
                try:
                    self.action_callback(action)
                except Exception as e:
                    logger.error(f"Action callback failed: {e}")

            logger.info(
                f"News action: {action.action} for {action.symbols} - {action.reason}"
            )

        return action

    def _analyze_news(self, news: NewsEvent) -> NewsEvent:
        """Enhance news with sentiment and urgency analysis."""
        headline_lower = news.headline.lower()
        content_lower = news.raw_content.lower() if news.raw_content else headline_lower

        # Check critical keywords
        for keyword, urgency in self.CRITICAL_KEYWORDS.items():
            if keyword in headline_lower or keyword in content_lower:
                news.urgency = max(news.urgency, urgency)

        # Calculate sentiment from keywords
        sentiment_score = 0.0
        keyword_count = 0

        for keyword, score in self.POSITIVE_KEYWORDS.items():
            if keyword in headline_lower:
                sentiment_score += score
                keyword_count += 1

        for keyword, score in self.NEGATIVE_KEYWORDS.items():
            if keyword in headline_lower:
                sentiment_score += score
                keyword_count += 1

        if keyword_count > 0:
            news.sentiment = max(-1, min(1, sentiment_score))

        # Boost urgency for negative high-sentiment news
        if news.sentiment < -0.5:
            news.urgency = max(news.urgency, 7)

        return news

    def _generate_action(self, news: NewsEvent) -> NewsAction:
        """Generate action based on news analysis."""
        action = "NONE"
        reason = news.headline[:100]

        # Critical events - close positions
        if news.is_critical and news.sentiment < -0.3:
            action = "CLOSE_ALL" if len(news.symbols) == 0 else "CLOSE_SYMBOL"
            reason = f"CRITICAL: {news.headline[:80]}"

        # High urgency negative - hedge or reduce
        elif news.urgency >= 7 and news.sentiment < -0.4:
            action = "HEDGE"
            reason = f"Negative news: {news.headline[:80]}"

        # Medium urgency negative - reduce size
        elif news.urgency >= 5 and news.sentiment < -0.3:
            action = "REDUCE_SIZE"
            reason = f"Caution: {news.headline[:80]}"

        # Any actionable news - alert
        elif news.is_actionable:
            action = "ALERT"
            reason = news.headline[:100]

        return NewsAction(
            action=action,
            symbols=news.symbols if news.symbols else ["ALL"],
            urgency=news.urgency,
            reason=reason,
            sentiment=news.sentiment,
        )

    def _is_rate_limited(self) -> bool:
        """Check if processing is rate limited."""
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_count = sum(
            1 for e in self._recent_events
            if e.timestamp > one_minute_ago
        )
        return recent_count >= self.max_events_per_minute

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self._last_action_time:
            return False

        elapsed = (datetime.now() - self._last_action_time[symbol]).total_seconds()
        return elapsed < self.cooldown_seconds

    def create_event_from_text(
        self,
        headline: str,
        source: str = "unknown",
        symbols: Optional[List[str]] = None,
    ) -> NewsEvent:
        """Create a NewsEvent from text."""
        # Extract symbols from headline if not provided
        if symbols is None:
            symbols = self._extract_symbols(headline)

        return NewsEvent(
            id=f"{datetime.now().timestamp()}_{hash(headline)}",
            timestamp=datetime.now(),
            headline=headline,
            source=source,
            symbols=symbols,
            sentiment=0.0,
            urgency=5,
            category="unknown",
        )

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract trading symbols from text."""
        symbols = []
        text_upper = text.upper()

        # Common crypto symbols
        crypto_symbols = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX"]
        for sym in crypto_symbols:
            if sym in text_upper:
                symbols.append(f"{sym}/USDT")

        # Stock tickers (simple pattern)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        for match in re.findall(ticker_pattern, text):
            if match in stock_symbols:
                symbols.append(match)

        return symbols

    def get_recent_events(self, minutes: int = 60) -> List[NewsEvent]:
        """Get events from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [e for e in self._recent_events if e.timestamp > cutoff]

    def get_stats(self) -> Dict:
        """Get reactor statistics."""
        recent = self.get_recent_events(60)
        return {
            "events_processed": self._events_processed,
            "actions_triggered": self._actions_triggered,
            "events_last_hour": len(recent),
            "critical_events_last_hour": sum(1 for e in recent if e.is_critical),
            "avg_sentiment_last_hour": (
                sum(e.sentiment for e in recent) / len(recent) if recent else 0
            ),
        }


# Singleton
_news_reactor: Optional[RealTimeNewsReactor] = None


def get_news_reactor(
    action_callback: Optional[Callable[[NewsAction], None]] = None
) -> RealTimeNewsReactor:
    """Get or create the RealTimeNewsReactor singleton."""
    global _news_reactor
    if _news_reactor is None:
        _news_reactor = RealTimeNewsReactor(action_callback=action_callback)
    return _news_reactor
