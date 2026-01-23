"""
News Reasoner - News Sentiment Analysis for Trading.

Analyzes news sentiment to inform trading decisions:
- Fetches news from free sources (CryptoPanic, Yahoo, RSS)
- Rule-based sentiment for routine analysis
- LLM analysis for important news
- Integrates with trading signals
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """A single news item."""

    title: str
    source: str
    url: str = ""
    published_at: Optional[datetime] = None
    sentiment: float = 0.0  # -1 to 1
    urgency: float = 0.0  # 0 to 1
    relevance: float = 0.0  # 0 to 1
    symbols: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "relevance": self.relevance,
            "symbols": self.symbols,
            "summary": self.summary,
        }


@dataclass
class NewsContext:
    """News context for trading decisions."""

    overall_sentiment: float  # -1 to 1
    sentiment_label: str  # "very_negative", "negative", "neutral", "positive", "very_positive"
    confidence: float
    news_count: int
    top_news: List[NewsItem]
    breaking_news: List[NewsItem]
    sentiment_trend: str  # "improving", "stable", "deteriorating"
    trading_impact: str  # "bullish", "neutral", "bearish"
    summary: str
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "overall_sentiment": self.overall_sentiment,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "news_count": self.news_count,
            "top_news": [n.to_dict() for n in self.top_news[:5]],
            "breaking_news": [n.to_dict() for n in self.breaking_news],
            "sentiment_trend": self.sentiment_trend,
            "trading_impact": self.trading_impact,
            "summary": self.summary,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


class NewsReasoner:
    """
    Analyzes news for trading decisions.

    Free sources:
    1. CryptoPanic API (crypto news)
    2. Yahoo Finance news
    3. RSS feeds

    Premium sources (optional - set API keys in .env):
    4. NewsAPI (newsapi.org) - 100 free requests/day
    5. Alpha Vantage News - 5 free requests/min
    6. Finnhub (finnhub.io) - 60 free requests/min

    Tiered analysis:
    - Routine: Rule-based sentiment (free, instant)
    - Important: Ollama analysis (free, local)
    - Critical: Claude analysis (paid, high quality)
    """

    # Premium news source configurations
    PREMIUM_SOURCES = {
        "newsapi": {
            "base_url": "https://newsapi.org/v2/everything",
            "env_key": "NEWSAPI_API_KEY",
            "rate_limit": 100,  # per day for free tier
            "categories": ["business", "technology"],
        },
        "alphavantage": {
            "base_url": "https://www.alphavantage.co/query",
            "env_key": "ALPHAVANTAGE_API_KEY",
            "rate_limit": 5,  # per minute
            "function": "NEWS_SENTIMENT",
        },
        "finnhub": {
            "base_url": "https://finnhub.io/api/v1/news",
            "env_key": "FINNHUB_API_KEY",
            "rate_limit": 60,  # per minute
            "categories": ["general", "forex", "crypto"],
        },
    }

    # Keywords for rule-based sentiment
    POSITIVE_KEYWORDS = [
        "surge",
        "soar",
        "rally",
        "bullish",
        "breakout",
        "all-time high",
        "adoption",
        "partnership",
        "approval",
        "upgrade",
        "growth",
        "record",
        "breakthrough",
        "milestone",
        "inflow",
        "accumulation",
    ]

    NEGATIVE_KEYWORDS = [
        "crash",
        "plunge",
        "dump",
        "bearish",
        "breakdown",
        "hack",
        "ban",
        "lawsuit",
        "sec",
        "investigation",
        "fraud",
        "scam",
        "outflow",
        "sell-off",
        "warning",
        "risk",
        "fear",
        "panic",
    ]

    URGENT_KEYWORDS = [
        "breaking",
        "urgent",
        "alert",
        "just in",
        "now",
        "flash",
        "emergency",
        "crash",
        "halt",
        "suspend",
    ]

    # Symbol mappings for relevance
    SYMBOL_KEYWORDS = {
        "BTC": ["bitcoin", "btc", "₿"],
        "ETH": ["ethereum", "eth", "ether"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
        "DOGE": ["dogecoin", "doge"],
        "AAPL": ["apple", "aapl", "iphone", "mac"],
        "MSFT": ["microsoft", "msft", "azure", "windows"],
        "GOOGL": ["google", "alphabet", "googl", "goog"],
        "NVDA": ["nvidia", "nvda", "gpu", "ai chip"],
        "AMZN": ["amazon", "amzn", "aws"],
        "GOLD": ["gold", "xau", "bullion"],
        "OIL": ["oil", "crude", "brent", "wti"],
    }

    def __init__(
        self,
        llm_router=None,
        cryptopanic_token: Optional[str] = None,
        cache_duration_minutes: int = 5,
        enable_premium: bool = True,
    ):
        """
        Initialize the News Reasoner.

        Args:
            llm_router: LLM router for advanced analysis
            cryptopanic_token: CryptoPanic API token (optional)
            cache_duration_minutes: How long to cache news
            enable_premium: Whether to use premium sources if API keys are available
        """
        self.llm_router = llm_router
        self.cryptopanic_token = cryptopanic_token or os.getenv("CRYPTOPANIC_API_KEY")
        self.cache_duration = cache_duration_minutes
        self.enable_premium = enable_premium

        # News cache
        self._news_cache: Dict[str, Tuple[List[NewsItem], datetime]] = {}
        self._sentiment_history: List[float] = []

        # Premium API keys (loaded from environment)
        self._premium_keys = {}
        if enable_premium:
            for source, config in self.PREMIUM_SOURCES.items():
                key = os.getenv(config["env_key"])
                if key:
                    self._premium_keys[source] = key
                    logger.info(f"Premium news source enabled: {source}")

        logger.info(f"News Reasoner initialized (premium sources: {len(self._premium_keys)})")

    def get_available_sources(self) -> Dict[str, bool]:
        """Get availability status of all news sources."""
        return {
            "cryptopanic": bool(self.cryptopanic_token),
            "yahoo_finance": True,  # Always available
            "newsapi": "newsapi" in self._premium_keys,
            "alphavantage": "alphavantage" in self._premium_keys,
            "finnhub": "finnhub" in self._premium_keys,
        }

    def _fetch_newsapi(self, query: str, hours_back: int = 24) -> List[NewsItem]:
        """Fetch news from NewsAPI (premium)."""
        if "newsapi" not in self._premium_keys:
            return []

        try:
            import requests

            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%d")

            response = requests.get(
                self.PREMIUM_SOURCES["newsapi"]["base_url"],
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "language": "en",
                    "apiKey": self._premium_keys["newsapi"],
                },
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(f"NewsAPI error: {response.status_code}")
                return []

            data = response.json()
            articles = data.get("articles", [])

            items = []
            for article in articles[:10]:  # Limit to 10
                item = NewsItem(
                    title=article.get("title", ""),
                    source="newsapi:" + article.get("source", {}).get("name", "unknown"),
                    url=article.get("url", ""),
                    published_at=datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    )
                    if article.get("publishedAt")
                    else None,
                    summary=article.get("description", ""),
                )
                item.sentiment = self._analyze_sentiment_rule_based(item.title)
                items.append(item)

            return items

        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    def _fetch_finnhub(self, category: str = "general") -> List[NewsItem]:
        """Fetch news from Finnhub (premium)."""
        if "finnhub" not in self._premium_keys:
            return []

        try:
            import requests

            response = requests.get(
                self.PREMIUM_SOURCES["finnhub"]["base_url"],
                params={
                    "category": category,
                    "token": self._premium_keys["finnhub"],
                },
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(f"Finnhub error: {response.status_code}")
                return []

            articles = response.json()

            items = []
            for article in articles[:10]:
                item = NewsItem(
                    title=article.get("headline", ""),
                    source="finnhub:" + article.get("source", "unknown"),
                    url=article.get("url", ""),
                    published_at=datetime.fromtimestamp(article["datetime"])
                    if article.get("datetime")
                    else None,
                    summary=article.get("summary", ""),
                )
                item.sentiment = self._analyze_sentiment_rule_based(item.title)
                items.append(item)

            return items

        except Exception as e:
            logger.warning(f"Finnhub fetch failed: {e}")
            return []

    def _fetch_alphavantage_sentiment(self, tickers: str) -> List[NewsItem]:
        """Fetch news sentiment from Alpha Vantage (premium)."""
        if "alphavantage" not in self._premium_keys:
            return []

        try:
            import requests

            response = requests.get(
                self.PREMIUM_SOURCES["alphavantage"]["base_url"],
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": tickers,
                    "apikey": self._premium_keys["alphavantage"],
                },
                timeout=15,
            )

            if response.status_code != 200:
                logger.warning(f"Alpha Vantage error: {response.status_code}")
                return []

            data = response.json()
            feed = data.get("feed", [])

            items = []
            for article in feed[:10]:
                # Alpha Vantage provides sentiment scores directly
                sentiment_score = float(article.get("overall_sentiment_score", 0))

                item = NewsItem(
                    title=article.get("title", ""),
                    source="alphavantage:" + ", ".join(article.get("authors", ["unknown"])),
                    url=article.get("url", ""),
                    published_at=datetime.strptime(article["time_published"], "%Y%m%dT%H%M%S")
                    if article.get("time_published")
                    else None,
                    sentiment=sentiment_score,
                    summary=article.get("summary", ""),
                )

                # Get relevance from ticker sentiment
                ticker_sentiments = article.get("ticker_sentiment", [])
                if ticker_sentiments:
                    item.relevance = float(ticker_sentiments[0].get("relevance_score", 0))

                items.append(item)

            return items

        except Exception as e:
            logger.warning(f"Alpha Vantage fetch failed: {e}")
            return []

    def get_news_context(
        self,
        symbols: Optional[List[str]] = None,
        hours_lookback: int = 24,
    ) -> NewsContext:
        """
        Get news context for trading decisions.

        Args:
            symbols: Symbols to filter news for
            hours_lookback: Hours of news to analyze

        Returns:
            NewsContext with sentiment analysis
        """
        symbols = symbols or []

        # Fetch news from all sources
        all_news = self._fetch_all_news(symbols, hours_lookback)

        if not all_news:
            return NewsContext(
                overall_sentiment=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                news_count=0,
                top_news=[],
                breaking_news=[],
                sentiment_trend="stable",
                trading_impact="neutral",
                summary="No recent news available",
            )

        # Analyze sentiment
        for news in all_news:
            if news.sentiment == 0:
                news.sentiment = self._analyze_sentiment_rule_based(news.title)
            news.urgency = self._calculate_urgency(news.title)
            news.relevance = self._calculate_relevance(news.title, symbols)

        # Calculate overall sentiment (weighted by relevance and recency)
        overall_sentiment = self._calculate_overall_sentiment(all_news)
        sentiment_label = self._sentiment_to_label(overall_sentiment)

        # Track sentiment history
        self._sentiment_history.append(overall_sentiment)
        if len(self._sentiment_history) > 24:
            self._sentiment_history = self._sentiment_history[-24:]

        # Calculate trend
        sentiment_trend = self._calculate_sentiment_trend()

        # Get breaking news (high urgency)
        breaking = [n for n in all_news if n.urgency > 0.7]
        breaking.sort(key=lambda x: x.urgency, reverse=True)

        # Get top news (most relevant)
        top_news = sorted(all_news, key=lambda x: x.relevance, reverse=True)[:10]

        # Determine trading impact
        trading_impact = self._determine_trading_impact(overall_sentiment, breaking)

        # Generate summary
        summary = self._generate_summary(
            overall_sentiment, sentiment_trend, len(all_news), breaking
        )

        return NewsContext(
            overall_sentiment=overall_sentiment,
            sentiment_label=sentiment_label,
            confidence=min(0.9, 0.3 + len(all_news) * 0.05),  # More news = more confidence
            news_count=len(all_news),
            top_news=top_news,
            breaking_news=breaking[:5],
            sentiment_trend=sentiment_trend,
            trading_impact=trading_impact,
            summary=summary,
        )

    def get_confidence_modifier(
        self,
        symbols: List[str],
        base_confidence: float,
    ) -> Tuple[float, str]:
        """
        Get confidence modifier based on news sentiment.

        Args:
            symbols: Symbols being traded
            base_confidence: Base signal confidence

        Returns:
            Tuple of (modified_confidence, reasoning)
        """
        context = self.get_news_context(symbols, hours_lookback=4)

        if context.news_count == 0:
            return base_confidence, "No recent news"

        # Adjust confidence based on sentiment alignment
        # If bullish signal and positive news -> boost confidence
        # If bullish signal and negative news -> reduce confidence
        modifier = 1.0 + (context.overall_sentiment * 0.2)  # Max +/- 20%
        modifier = max(0.8, min(1.2, modifier))

        modified = base_confidence * modifier
        modified = max(0.0, min(1.0, modified))

        reasoning = f"News: {context.sentiment_label} ({context.overall_sentiment:+.2f}), {context.news_count} articles"

        if context.breaking_news:
            reasoning += f", {len(context.breaking_news)} breaking"

        return modified, reasoning

    def _fetch_all_news(
        self,
        symbols: List[str],
        hours_lookback: int,
    ) -> List[NewsItem]:
        """Fetch news from all available sources."""
        all_news = []

        # Check cache
        cache_key = f"{','.join(sorted(symbols))}:{hours_lookback}"
        if cache_key in self._news_cache:
            cached_news, cached_time = self._news_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_duration):
                return cached_news

        # Fetch from CryptoPanic (crypto)
        crypto_symbols = [s for s in symbols if s in ["BTC", "ETH", "SOL", "XRP", "DOGE"]]
        if crypto_symbols:
            try:
                crypto_news = self._fetch_cryptopanic(crypto_symbols)
                all_news.extend(crypto_news)
            except Exception as e:
                logger.debug(f"CryptoPanic fetch error: {e}")

        # Fetch from Finnhub (premium - works best)
        if "finnhub" in self._premium_keys:
            try:
                finnhub_news = self._fetch_finnhub("general")
                all_news.extend(finnhub_news)
                # Also fetch crypto category
                crypto_news = self._fetch_finnhub("crypto")
                all_news.extend(crypto_news)
            except Exception as e:
                logger.debug(f"Finnhub fetch error: {e}")

        # Fetch from Yahoo Finance (fallback)
        if not all_news:
            try:
                yahoo_news = self._fetch_yahoo_news(symbols)
                all_news.extend(yahoo_news)
            except Exception as e:
                logger.debug(f"Yahoo news fetch error: {e}")

        # Cache results
        self._news_cache[cache_key] = (all_news, datetime.now())

        return all_news

    def _fetch_cryptopanic(self, symbols: List[str]) -> List[NewsItem]:
        """Fetch news from CryptoPanic API."""
        if not self.cryptopanic_token:
            return []

        try:
            import requests

            currencies = ",".join([s.lower() for s in symbols])
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_token}&currencies={currencies}&filter=hot"

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()
            news_items = []

            for item in data.get("results", [])[:20]:
                news = NewsItem(
                    title=item.get("title", ""),
                    source="CryptoPanic",
                    url=item.get("url", ""),
                    published_at=datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
                    if item.get("published_at")
                    else None,
                    symbols=[c["code"] for c in item.get("currencies", [])],
                )
                news_items.append(news)

            return news_items

        except Exception as e:
            logger.warning(f"CryptoPanic error: {e}")
            return []

    def _fetch_yahoo_news(self, symbols: List[str]) -> List[NewsItem]:
        """Fetch news from Yahoo Finance."""
        try:
            import yfinance as yf

            news_items = []

            for symbol in symbols[:5]:  # Limit to avoid rate limits
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news

                    for item in news[:5]:
                        news_item = NewsItem(
                            title=item.get("title", ""),
                            source="Yahoo Finance",
                            url=item.get("link", ""),
                            published_at=datetime.fromtimestamp(item["providerPublishTime"])
                            if item.get("providerPublishTime")
                            else None,
                            symbols=[symbol],
                        )
                        news_items.append(news_item)

                    time.sleep(0.5)  # Rate limit protection

                except Exception:
                    continue

            return news_items

        except Exception as e:
            logger.warning(f"Yahoo news error: {e}")
            return []

    def _analyze_sentiment_rule_based(self, text: str) -> float:
        """Analyze sentiment using rule-based approach."""
        text_lower = text.lower()

        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        # Score from -1 to 1
        sentiment = (positive_count - negative_count) / total

        # Amplify strong signals
        if positive_count > 2 and negative_count == 0:
            sentiment = min(1.0, sentiment * 1.2)
        elif negative_count > 2 and positive_count == 0:
            sentiment = max(-1.0, sentiment * 1.2)

        return sentiment

    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score for news item."""
        text_lower = text.lower()

        urgent_count = sum(1 for kw in self.URGENT_KEYWORDS if kw in text_lower)

        # Urgency from 0 to 1
        return min(1.0, urgent_count * 0.3)

    def _calculate_relevance(self, text: str, symbols: List[str]) -> float:
        """Calculate relevance score for symbols."""
        if not symbols:
            return 0.5  # Default relevance

        text_lower = text.lower()
        relevant_count = 0

        for symbol in symbols:
            # Check symbol directly
            if symbol.lower() in text_lower:
                relevant_count += 1
                continue

            # Check keywords
            keywords = self.SYMBOL_KEYWORDS.get(symbol, [])
            if any(kw in text_lower for kw in keywords):
                relevant_count += 1

        return min(1.0, relevant_count / len(symbols))

    def _calculate_overall_sentiment(self, news_items: List[NewsItem]) -> float:
        """Calculate weighted overall sentiment."""
        if not news_items:
            return 0.0

        total_weight = 0.0
        weighted_sentiment = 0.0

        for news in news_items:
            # Weight by relevance and recency
            weight = news.relevance * 0.7 + (1 - news.urgency) * 0.3
            if news.urgency > 0.5:
                weight *= 1.5  # Boost urgent news

            weighted_sentiment += news.sentiment * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sentiment / total_weight

    def _sentiment_to_label(self, sentiment: float) -> str:
        """Convert sentiment score to label."""
        if sentiment > 0.5:
            return "very_positive"
        elif sentiment > 0.15:
            return "positive"
        elif sentiment < -0.5:
            return "very_negative"
        elif sentiment < -0.15:
            return "negative"
        return "neutral"

    def _calculate_sentiment_trend(self) -> str:
        """Calculate sentiment trend from history."""
        if len(self._sentiment_history) < 3:
            return "stable"

        recent = self._sentiment_history[-3:]
        older = (
            self._sentiment_history[-6:-3]
            if len(self._sentiment_history) >= 6
            else self._sentiment_history[:3]
        )

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg

        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "deteriorating"
        return "stable"

    def _determine_trading_impact(
        self,
        sentiment: float,
        breaking_news: List[NewsItem],
    ) -> str:
        """Determine trading impact from news."""
        # Check for breaking news override
        if breaking_news:
            breaking_sentiment = sum(n.sentiment for n in breaking_news) / len(breaking_news)
            if abs(breaking_sentiment) > 0.5:
                return "bullish" if breaking_sentiment > 0 else "bearish"

        if sentiment > 0.3:
            return "bullish"
        elif sentiment < -0.3:
            return "bearish"
        return "neutral"

    def _generate_summary(
        self,
        sentiment: float,
        trend: str,
        news_count: int,
        breaking: List[NewsItem],
    ) -> str:
        """Generate news summary."""
        sentiment_label = self._sentiment_to_label(sentiment)

        summary = f"News sentiment: {sentiment_label} ({sentiment:+.2f})"

        if trend != "stable":
            summary += f", {trend}"

        summary += f" based on {news_count} articles"

        if breaking:
            summary += f". {len(breaking)} breaking news items"
            if breaking[0].title:
                summary += f': "{breaking[0].title[:50]}..."'

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """Get reasoner summary."""
        return {
            "cached_queries": len(self._news_cache),
            "sentiment_history_length": len(self._sentiment_history),
            "current_sentiment": self._sentiment_history[-1] if self._sentiment_history else 0,
            "cryptopanic_enabled": bool(self.cryptopanic_token),
        }
